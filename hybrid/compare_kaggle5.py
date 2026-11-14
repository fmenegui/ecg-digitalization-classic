"""Paired frozen-transfer pilot. Reports every failure and missing prediction.

Synthetic images are a small transfer probe, not Kaggle leaderboard evaluation.
Run: python -m hybrid.compare_kaggle5 --source ../kaggle-5th --device mps
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import hashlib
import html
import json
import platform
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
from ecgdig import digitize,Config
from ecgdig.layouts import LEADS
from ecgdig.synth import synth_signals,render

CASES = [
    ('3x4+1','none',25,10,10), ('3x4+1','medium',25,10,10),
    ('6x2','none',25,10,10), ('6x2','medium',25,10,5),
    ('3x4+1-cabrera','none',25,10,10), ('3x4+1-cabrera','medium',25,10,10),
    ('12x1','none',25,5,5), ('12x1','medium',25,5,5),
    ('6x1-prec','none',50,10,10), ('6x1-prec','medium',50,10,10),
    ('3x4+3','none',25,10,10), ('3x4+3','medium',25,10,5),
]

def metrics(gt,pred,shown):
    valid=shown & np.isfinite(gt)
    have=valid & np.isfinite(pred)
    # Missing samples are penalized as zero; intersection-only score is also
    # saved, but never used to hide missing predictions or algorithm failures.
    filled=np.nan_to_num(pred,nan=0,posinf=0,neginf=0)
    power=float(np.sum(gt[valid]**2)); error=float(np.sum((gt[valid]-filled[valid])**2))
    snr=10*np.log10(max(power,1e-12)/max(error,1e-12))
    rmse=np.sqrt(error/max(1,valid.sum()))*1000
    pe=np.abs(gt[valid]); threshold=np.quantile(pe,.99) if pe.size else np.inf
    peaks=valid & (np.abs(gt)>=threshold)
    return dict(snr_db=float(snr),rmse_uv=float(rmse),coverage=float(have.sum()/max(1,valid.sum())),
                peak_rmse_uv=float(np.sqrt(np.mean((gt[peaks]-filled[peaks])**2))*1000) if peaks.any() else None)

def canonical(res,gt):
    out=np.full_like(gt,np.nan)
    for i,name in enumerate(res['leads']):
        if name in LEADS:
            n=min(gt.shape[1],res['signals'].shape[1]);out[LEADS.index(name),:n]=res['signals'][i,:n]
    return out

def metadata(res):
    keys=['layout','layout_rows','rotation','gains','duration_s','speed_mm_per_s','speed_source',
          'duration_assumed','horizontal_scale','ppmm','angle','pulses_found','pulses_right']
    return {k:res.get(k) for k in keys}

def generate(root,seed,limit):
    cases=[];root.mkdir(parents=True,exist_ok=True)
    for idx,(layout,level,speed,gl,gp) in enumerate(CASES[:limit]):
        rng=np.random.default_rng(seed+idx)
        gt=synth_signals(rng)
        img,shown,meta=render(gt,rng,ppmm=6.5,degrade=level,layout=layout,speed=speed,
                              gain_limb=gl,gain_prec=gp)
        name=f'{idx:02d}_{layout}_{level}';path=root/(name+'.png')
        cv2.imwrite(str(path),img)
        np.savez_compressed(root/(name+'.npz'),sig=gt,shown=shown)
        (root/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
        cases.append(dict(id=name,image=str(path),layout=layout,degrade=level,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=seed+idx))
    return cases

def run(a):
    import torch
    torch.set_num_threads(a.threads);cv2.setNumThreads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    cases=json.loads(manifest.read_text()) if manifest.exists() else generate(out/'inputs',a.seed,a.limit)
    manifest.write_text(json.dumps(cases,indent=2)+'\n')
    classic=Config()
    neural=replace(classic,waveform_model=a.model,waveform_source=a.source,waveform_device=a.device)
    records=[]
    # A checkpoint/config error must abort, not turn into an apparent fallback
    # win. Load once before any paired measurement.
    from ecgdig.kaggle5 import get_predictor
    predictor=get_predictor(a.model,a.source,a.device)
    predictor.progress=lambda info: print('  neural window',info,flush=True)
    for case in cases[:a.limit]:
        raw=np.load(Path(case['image']).with_suffix('.npz'));gt,shown=raw['sig'],raw['shown']
        outputs={}
        for variant,cfg in [('classic',classic),('kaggle5_frozen',neural)]:
            result_path=out/(case['id']+'_'+variant+'.json')
            pred_path=result_path.with_suffix('.npz')
            if result_path.exists() and pred_path.exists():
                record=json.loads(result_path.read_text());pred=np.load(pred_path)['signals']
            else:
                start=time.perf_counter()
                try:
                    res=digitize(case['image'],cfg)
                    pred=canonical(res,gt)
                    record=dict(id=case['id'],format=case['layout'],degrade=case['degrade'],variant=variant,
                                status='ok',seconds=time.perf_counter()-start,metadata=metadata(res),
                                hybrid=res['hybrid'],**metrics(gt,pred,shown))
                except Exception as ex:
                    pred=np.full_like(gt,np.nan)
                    record=dict(id=case['id'],format=case['layout'],degrade=case['degrade'],variant=variant,
                                status='failed',error=str(ex),traceback=traceback.format_exc(),
                                seconds=time.perf_counter()-start,metadata=None,**metrics(gt,pred,shown))
                np.savez_compressed(pred_path,signals=pred)
                result_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            records.append(record);outputs[variant]=record
            print(case['id'],variant,record['status'],f"SNR={record['snr_db']:.2f}",f"{record['seconds']:.1f}s",flush=True)
        c,h=outputs['classic'],outputs['kaggle5_frozen']
        if c['status']==h['status']=='ok' and c['metadata']!=h['metadata']:
            raise AssertionError('Neural backend changed structural metadata')
        report(records,out,a)

def report(records,out,a):
    keys=['id','format','degrade','variant','status','snr_db','rmse_uv','peak_rmse_uv','coverage','seconds']
    with (out/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(records)
    pairs=[]
    for ident in dict.fromkeys(r['id'] for r in records):
        rr={r['variant']:r for r in records if r['id']==ident}
        if len(rr)<2:continue
        c,h=rr['classic'],rr['kaggle5_frozen']
        pairs.append(dict(id=ident,format=c['format'],degrade=c['degrade'],classic=c['snr_db'],
                          hybrid=h['snr_db'],delta=h['snr_db']-c['snr_db'],
                          structure_equal=c['metadata']==h['metadata'],status=h['status']))
    formats=[]
    for fmt in dict.fromkeys(p['format'] for p in pairs):
        pp=[p for p in pairs if p['format']==fmt]
        formats.append(dict(format=fmt,n=len(pp),classic=float(np.mean([p['classic'] for p in pp])),
                            hybrid=float(np.mean([p['hybrid'] for p in pp])),delta=float(np.mean([p['delta'] for p in pp]))))
    with (out/'paired.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,['id','format','degrade','classic','hybrid','delta','structure_equal','status'])
        w.writeheader();w.writerows(pairs)
    import torch
    summary=dict(protocol='synthetic transfer pilot; zero-fill missing; no post-hoc alignment; pooled SNR per image',
                 environment=dict(python=platform.python_version(),platform=platform.platform(),
                                  torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,device=a.device),
                 seed=a.seed,pairs=pairs,formats=formats,model=json.loads((Path(a.model)/'provenance.json').read_text()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'per_format.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,['format','n','classic','hybrid','delta']);w.writeheader();w.writerows(formats)
    rows=''.join(f"<tr><td>{html.escape(p['id'])}</td><td>{p['classic']:.2f}</td><td>{p['hybrid']:.2f}</td><td>{p['delta']:+.2f}</td><td>{p['structure_equal']}</td></tr>" for p in pairs)
    fmtrows=''.join(f"<tr><td>{p['format']}</td><td>{p['n']}</td><td>{p['classic']:.2f}</td><td>{p['hybrid']:.2f}</td><td>{p['delta']:+.2f}</td></tr>" for p in formats)
    page=f'''<!doctype html><meta charset="utf-8"><title>ecgdig: clássico × Kaggle 5º</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#18243c}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}td:first-child{{text-align:left}}table{{border-collapse:collapse;width:100%}}h1{{font-size:30px}}.note{{background:#edf3fa;padding:20px}}</style>
<h1>Clássico × pesos congelados do 5º colocado</h1><p class="note">Piloto sintético multiformato. Geometria, layout, ganho e tempo vêm do clássico. Rede CoaT + VGG19, EMA 0,999; quarto canal = grade anônima. Sem ajuste de pesos. Não é reprodução do Kaggle nem validação clínica. SNR agrupado por imagem, sem alinhamento posterior; amostras ausentes penalizadas como zero.</p>
<h2>Por formato</h2><table><tr><th>Formato</th><th>N</th><th>Clássico (dB)</th><th>Híbrido (dB)</th><th>Δ (dB)</th></tr>{fmtrows}</table>
<h2>Por imagem</h2><table><tr><th>Imagem</th><th>Clássico (dB)</th><th>Híbrido (dB)</th><th>Δ (dB)</th><th>Estrutura idêntica</th></tr>{rows}</table>
<p>Consulte per_image.csv para cobertura, RMSE em µV, erro nos 1% de maiores amplitudes e tempo. O protocolo não escolhe vencedor por imagem usando o gabarito.</p>'''
    (out/'report.html').write_text(page)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--model',default='models/kaggle5')
    p.add_argument('--out',default='results/kaggle5-pilot');p.add_argument('--device',default='cpu')
    p.add_argument('--limit',type=int,default=len(CASES));p.add_argument('--seed',type=int,default=20261114)
    p.add_argument('--threads',type=int,default=2);run(p.parse_args())
