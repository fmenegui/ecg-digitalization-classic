"""Frozen 5th-place waveform model on layout-independent local windows.

This is a transfer experiment, not a reproduction of the Kaggle submission.
The 4th channel is an anonymous nominal grid, without lead names or a page
template. Layout, baseline, paper speed and gain come from the classic pipeline.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import sys
import cv2
import numpy as np

def crop_coordinates(start, baseline, px_per_mv, px_per_sec):
    """Match the official 1024x1024 / 1000-column codec, in source pixels."""
    x = start + (np.arange(1024,dtype=np.float32)+0.5-12)*px_per_sec/400
    y = baseline + (np.arange(1024,dtype=np.float32)+0.5-512)*px_per_mv/160
    return np.broadcast_to(x,(1024,1024)).astype(np.float32), np.broadcast_to(y[:,None],(1024,1024)).astype(np.float32)

def anonymous_template():
    """Canonical 10 mm/mV, 25 mm/s blank grid; no learned page identity."""
    a=np.full((1024,1024),255,np.uint8)
    for p in range(0,1024,16):
        color=180 if p%80 == 0 else 225
        cv2.line(a,(p,0),(p,1023),color,1)
        cv2.line(a,(0,p),(1023,p),color,1)
    return a

def tiled_local_head(head,x,tile=64,halo=2):
    import torch
    pieces=[]
    for top in range(0,x.shape[2],tile):
        bottom=min(x.shape[2],top+tile)
        lo=max(0,top-halo);hi=min(x.shape[2],bottom+halo)
        pieces.append(head(x[:,:,lo:hi])[:,:,top-lo:bottom-lo])
    return torch.cat(pieces,dim=2)

def tiled_decoder_block(block,feature,height,width,skip,tile=64):
    """Exact nearest upsampling + local two-convolution block with a 2px halo.

    Valid only for the selected checkpoint (no spatial attention; channelwise
    LayerNorm). Global encoder attention is never tiled.
    """
    import torch
    from torch.nn import functional as F
    horizontal=F.interpolate(feature,size=(feature.shape[2],width),mode='nearest')
    output=None
    for top in range(0,height,tile):
        bottom=min(height,top+tile);lo=max(0,top-2);hi=min(height,bottom+2)
        idx=torch.arange(lo,hi,device=feature.device)*feature.shape[2]//height
        up=horizontal.index_select(2,idx)
        if skip is not None:
            cat=torch.cat([up,block.skip_norm(skip[:,:,lo:hi])],dim=1)
            cat=block.attention1(block.cat_act(block.cat_norm(cat)))
        else:
            cat=up
        result=block.attention2(block.conv2(block.conv1(cat)))
        if block.use_residual: result=result+block.res_conv(cat)
        if output is None:
            output=result.new_empty((result.shape[0],result.shape[1],height,width))
        output[:,:,top:bottom]=result[:,:,top-lo:bottom-lo]
    return output

@lru_cache(maxsize=2)
def get_predictor(model_dir, source_dir, device='cpu'):
    return Kaggle5(model_dir, source_dir, device)

class Kaggle5:
    def __init__(self, model_dir, source_dir, device='cpu'):
        import torch
        from omegaconf import OmegaConf
        source=Path(source_dir).resolve(); folder=Path(model_dir)
        if not (source/'src/ecg/model/signal_fine_heatmap_model.py').is_file():
            raise FileNotFoundError('waveform_source must point to kaggle_ecg_digitization checkout')
        for p in [source/'src',source/'yagm/src']:
            if str(p) not in sys.path: sys.path.insert(0,str(p))
        from ecg.model.signal_fine_heatmap_model import SignalFineHeatmapModel
        cfg=OmegaConf.load(folder/'config.yaml')
        # Full task checkpoint supplies every parameter; no ImageNet download.
        cfg.model.encoder.pretrained=False
        cfg.model.fine_encoder.pretrained=False
        self.model=SignalFineHeatmapModel(cfg)
        state=torch.load(folder/'ema.pt',map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True)
        self.device=torch.device(device)
        # The stride-1 decoder has large feature maps. FP16 bounds memory on
        # 8 GB Macs; record precision explicitly in every segment result.
        self.dtype=torch.float16
        self.model.to(device=self.device,dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        if self.device.type != 'cuda':
            # Head consists only of local convolutions and per-pixel channel norms.
            # Evaluate overlapping strips to avoid a full 384x1024x1024 temporary;
            # encoder/attention and the decoder still see the complete window.
            head=self.model.seg_head
            class TiledHead(torch.nn.Module):
                def __init__(self, head):
                    super().__init__();self.head=head
                def forward(self,x):
                    return tiled_local_head(self.head,x)
            self.model.seg_head=TiledHead(head)
            if cfg.model.decoder.attn_type is not None or cfg.model.decoder.norm_layer != 'layernorm' or cfg.model.decoder.interpolation_mode != 'nearest':
                raise ValueError('Tiled decoder requires the published local LayerNorm/nearest configuration')
            class TiledBlock(torch.nn.Module):
                def __init__(self,block):
                    super().__init__();self.block=block
                def forward(self,x,height,width,skip_connection=None):
                    if height < 256:
                        return self.block(x,height,width,skip_connection=skip_connection)
                    return tiled_decoder_block(self.block,x,height,width,skip_connection)
            self.model.decoder.blocks=torch.nn.ModuleList([TiledBlock(b) for b in self.model.decoder.blocks])
            decoder=self.model.decoder
            class FinestDecoder(torch.nn.Module):
                def __init__(self,decoder):
                    super().__init__();self.decoder=decoder
                def forward(self,features):
                    fs=features[::-1];shapes=[f.shape[2:] for f in fs]
                    skip=fs[1:] if self.decoder.skip_on_finest else fs[1:-1]
                    x=self.decoder.center(fs[0])
                    for i,block in enumerate(self.decoder.blocks):
                        h,w=shapes[i+1] if i+1<len(shapes) else (2*h,2*w)
                        x=block(x,h,w,skip_connection=skip[i] if i<len(skip) else None)
                    # Only finest scale is consumed by SignalFineHeatmapModel.
                    return [x]
            self.model.decoder=FinestDecoder(decoder)
        self.template=anonymous_template()
        self.window_seconds=2.5
        self.progress=None

    def predict(self, image, xn, baseline, px_per_mv, px_per_sec):
        import torch
        if px_per_mv <= 0 or px_per_sec <= 0 or len(xn) < 2:
            raise ValueError('Invalid classic calibration or segment')
        width=self.window_seconds*px_per_sec
        positions=[]; coords=[]; edge_count=0; windows=0
        # Each window preserves the measured physical scales. No fixed number
        # of rows, leads or columns; long rhythm strips are split automatically.
        for start in np.arange(xn[0],xn[-1]+1e-6,width):
            import time
            clock=time.perf_counter()
            mx,my=crop_coordinates(start,baseline,px_per_mv,px_per_sec)
            crop=cv2.remap(image,mx,my,cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
            four=np.concatenate([crop[...,::-1],self.template[...,None]],axis=-1)
            tensor=torch.from_numpy(four.transpose(2,0,1).copy()).unsqueeze(0).to(self.device,dtype=self.dtype)
            with torch.inference_mode():
                logits,_,_=self.model(tensor)
                # Official submission decodes by argmax (+ half-pixel), not
                # DSNT; keep this choice to isolate transfer from codec changes.
                yi=logits[0,0,:,12:1012].argmax(dim=0).cpu().numpy()
            xs=start+(np.arange(1000)+0.5)*px_per_sec/400
            ys=baseline+(yi+0.5-512)*px_per_mv/160
            edge_count+=int(((yi<8)|(yi>1015)).sum());windows+=1
            positions.append(xs);coords.append(ys)
            if self.progress is not None:
                self.progress(dict(window=windows,seconds=time.perf_counter()-clock))
            del logits,tensor
            if self.device.type=='mps': torch.mps.empty_cache()
        xx=np.concatenate(positions);yy=np.concatenate(coords)
        out=np.interp(xn,xx,yy)
        return out,dict(windows=windows,range_edge_fraction=edge_count/(1000*windows),
                        template='anonymous_nominal_grid',decoder='official_argmax',device=str(self.device),precision='float16',local_tiles=self.device.type!='cuda')
