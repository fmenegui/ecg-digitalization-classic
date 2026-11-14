"""Extract only EMA inference tensors from the official checkpoint over HTTP ranges.

The full Lightning checkpoint contains three copies of the model. No checkpoint
pickle code is executed: tensor rebuilds become descriptors; other globals are
inert placeholders. ZIP CRCs are checked before exporting a plain state_dict.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import urllib.request
import zipfile
import zlib

FILE = 'FINAL_01-19__21-15-59.746642_ALLDATA_FINAL_COAT512_VGG1024_GT1000_LR2en4_SEED980611_ep3_step70007_val_SNR28.006985.ckpt'
CONFIG = 'FINAL_01-19__21-15-59.746642_ALLDATA_FINAL_COAT512_VGG1024_GT1000_LR2en4_SEED980611_config.yaml'
BASE = 'https://www.kaggle.com/api/v1/datasets/download/dangnh0611/ecg-checkpoints/'

class Remote(io.RawIOBase):
    def __init__(self, url):
        headers = subprocess.check_output(['curl','-sS','--fail','--max-time','60','-D','-','-o','/dev/null','-r','0-0',url]).decode()
        self.url = next(s.split(': ',1)[1].strip() for s in headers.splitlines() if s.lower().startswith('location:'))
        with urllib.request.urlopen(urllib.request.Request(self.url,headers={'Range':'bytes=0-0'}), timeout=60) as r:
            self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if n <= 0: return b''
        start = self.pos
        req = urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{start+n-1}'})
        with urllib.request.urlopen(req, timeout=180) as r:
            if r.status != 206 or not r.headers['Content-Range'].startswith(f'bytes {start}-'):
                raise RuntimeError('Server did not honor byte range')
            data = r.read()
        if len(data) != n: raise IOError('Incomplete range')
        self.pos += n
        return data

class Node:
    def __init__(self, *args, **kwargs): self.args = args
    def __setstate__(self, state): pass
    def __setitem__(self, key, value): pass
    def append(self, value): pass
    def extend(self, value): pass

class DescriptorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict': return OrderedDict
        if module == 'torch' and name.endswith('Storage'): return name
        if module == 'torch._utils' and name.startswith('_rebuild_tensor'): return lambda *a: ('tensor', a)
        return Node
    def persistent_load(self, pid): return pid

def extract_ema(remote, out, source_bytes):
    import torch
    z = zipfile.ZipFile(remote)
    pkl = next(n for n in z.namelist() if n.endswith('/data.pkl'))
    obj = DescriptorUnpickler(io.BytesIO(z.read(pkl))).load()
    decay = 0.999
    state = obj['ema_state_dict'][decay]
    prefix = pkl[:-len('data.pkl')] + 'data/'
    selected = {k[6:]:v for k,v in state.items() if k.startswith('model.')}
    if not selected: raise ValueError('No model EMA tensors')
    infos = {v[1][0][2]:z.getinfo(prefix+str(v[1][0][2])) for v in selected.values()}
    start = min(i.header_offset for i in infos.values())
    end = max(i.header_offset+30+len(i.filename.encode())+65536+i.compress_size for i in infos.values())
    end = min(end,source_bytes)
    print(f'Downloading EMA {decay}: {len(selected)} tensors, {(end-start)/1e6:.1f} MB',flush=True)
    remote.seek(start); block = remote.read(end-start)
    arrays = {}
    for key,info in infos.items():
        off = info.header_offset-start
        name_len = int.from_bytes(block[off+26:off+28],'little')
        extra_len = int.from_bytes(block[off+28:off+30],'little')
        raw = block[off+30+name_len+extra_len:off+30+name_len+extra_len+info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED: raw = zlib.decompress(raw,-15)
        elif info.compress_type != zipfile.ZIP_STORED: raise ValueError('Unsupported ZIP compression')
        if zlib.crc32(raw) & 0xffffffff != info.CRC: raise ValueError('Tensor CRC mismatch')
        arrays[key] = raw
    dtype_map = {'FloatStorage':torch.float32,'LongStorage':torch.int64,'HalfStorage':torch.float16,'DoubleStorage':torch.float64,'ByteStorage':torch.uint8,'BoolStorage':torch.bool}
    result = {}
    for name,desc in selected.items():
        storage,offset,shape,stride,*_ = desc[1]
        tensor = torch.frombuffer(bytearray(arrays[storage[2]]), dtype=dtype_map[storage[1]])
        result[name] = torch.as_strided(tensor,shape,stride,offset).clone()
    target=out/'ema.pt'
    temporary=out/'ema.part.pt'
    torch.save(result,temporary)
    temporary.replace(target)
    with target.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'source':BASE+FILE,'source_bytes':source_bytes,'ema_decay':decay,'tensors':len(result),'sha256':digest,'upstream_commit':'8358d5b09a5b9519275e9b54c11716da5752a9d6'}

def curl_file(url, target, timeout=900):
    command=['curl','--fail','--location','--silent','--show-error',
             '--retry','3','--retry-delay','2','--connect-timeout','30',
             '--max-time',str(timeout),'--output',str(target),url]
    process=subprocess.run(command,capture_output=True,text=True)
    if process.returncode:
        raise RuntimeError(f'curl terminou com código {process.returncode}: {process.stderr.strip()}')

def download(out, mode='auto'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=None
    if mode in ('auto','range'):
        try:
            remote=Remote(BASE+FILE)
            meta=extract_ema(remote,out,remote.size)
            meta['download_mode']='range'
        except Exception as error:
            if mode=='range': raise
            print(f'Download parcial indisponível ({type(error).__name__}: {error}).',flush=True)
            print('Alternativa: baixar checkpoint completo (~941 MB), sem depender de HTTP Range.',flush=True)
    if meta is None:
        archive=out/'source.ckpt'
        partial=out/'source.ckpt.part'
        try:
            if not archive.exists():
                curl_file(BASE+FILE,partial)
                if not zipfile.is_zipfile(partial):
                    raise RuntimeError('Kaggle não retornou um checkpoint ZIP válido. Verifique a mensagem HTTP acima.')
                partial.replace(archive)
            with archive.open('rb') as stream:
                meta=extract_ema(stream,out,archive.stat().st_size)
            meta['download_mode']='full'
        finally:
            # Only remove temporary weights downloaded by this script.
            archive.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
    config_tmp=out/'config.yaml.part'
    curl_file(BASE+CONFIG,config_tmp,timeout=120)
    from omegaconf import OmegaConf
    config=OmegaConf.load(config_tmp)
    if OmegaConf.select(config,'model._target_') != 'ecg.model.signal_fine_heatmap_model.SignalFineHeatmapModel':
        raise RuntimeError('A configuração baixada não corresponde ao modelo esperado; pode ser uma página de erro.')
    config_tmp.replace(out/'config.yaml')
    (out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)
    return meta

if __name__ == '__main__':
    import sys
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='models/kaggle5')
    p.add_argument('--mode',choices=['auto','range','full'],default='auto')
    a=p.parse_args()
    try:
        download(a.out,a.mode)
    except Exception as error:
        import traceback
        print(f'FALHA NO DOWNLOAD: {type(error).__name__}: {error}',file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.exit(1)
