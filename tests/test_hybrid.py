"""Contract tests: neural assistance cannot change page interpretation."""
import importlib
import numpy as np
import pytest
from ecgdig import Config,digitize
from ecgdig.kaggle5 import crop_coordinates
from ecgdig.synth import render,synth_signals

def test_crop_codec_scales():
    mx,my=crop_coordinates(np.float64(20),np.float64(500),np.float64(100),np.float64(250))
    assert mx.dtype==my.dtype==np.float32
    assert mx.shape==my.shape==(1024,1024)
    assert mx[0,1012]-mx[0,12]==pytest.approx(625)
    assert my[672,0]-my[512,0]==pytest.approx(100)
    assert (my[511,0]+my[512,0])/2==pytest.approx(500)

def test_tiled_head_preserves_seams():
    torch=pytest.importorskip('torch')
    from ecgdig.kaggle5 import tiled_local_head
    torch.manual_seed(7)
    head=torch.nn.Sequential(torch.nn.Conv2d(3,8,3,padding=1),torch.nn.GELU(),torch.nn.Conv2d(8,1,3,padding=1)).eval()
    x=torch.randn(1,3,139,37)
    with torch.inference_mode():
        torch.testing.assert_close(tiled_local_head(head,x),head(x),atol=2e-6,rtol=2e-6)

def test_tiled_decoder_preserves_seams():
    torch=pytest.importorskip('torch')
    upstream=pytest.importorskip('yagm.models.unet_2d.decoder')
    from timm.layers import LayerNorm2d
    from ecgdig.kaggle5 import tiled_decoder_block
    torch.manual_seed(8)
    block=upstream.UnetDecoderBlock(3,2,8,norm_layer=LayerNorm2d,act_layer=torch.nn.GELU).eval()
    feature=torch.randn(1,3,70,37);skip=torch.randn(1,2,139,73)
    with torch.inference_mode():
        expected=block(feature,139,73,skip_connection=skip)
        actual=tiled_decoder_block(block,feature,139,73,skip,tile=32)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)

@pytest.fixture(scope='module')
def sample():
    rng=np.random.default_rng(412)
    image,_,_=render(synth_signals(rng),rng,ppmm=6,degrade='none')
    return image

def test_waveform_does_not_change_structure(sample,monkeypatch):
    class LocalPredictor:
        calls=0
        def predict(self,img,xn,baseline,px_per_mv,px_per_sec):
            self.calls+=1
            return baseline+np.sin((xn-xn[0])/px_per_sec*7)*px_per_mv,dict(windows=1)
    fake=LocalPredictor()
    monkeypatch.setattr('ecgdig.kaggle5.get_predictor',lambda *a:fake)
    classic=digitize(sample)
    hybrid=digitize(sample,Config(waveform_model='fake'))
    for key in ['layout_rows','gains','speed_mm_per_s','duration_s','rotation','pulses_found']:
        assert classic[key]==hybrid[key]
    assert fake.calls>0
    assert not np.allclose(classic['signals'],hybrid['signals'],equal_nan=True)
    assert hybrid['hybrid']['structural_map']=='classic'

def test_ink_map_is_used_after_layout(sample,monkeypatch):
    import ecgdig.inkmodel
    seen=[]
    module=importlib.import_module('ecgdig.digitize')
    original=module._LayoutCtx.choose
    def choose(self,*args,**kwargs):
        seen.append('layout')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(module._LayoutCtx,'choose',choose)
    # A no-op neural mixture must preserve every numerical output.
    def predict(image,path):
        seen.append('neural')
        return np.zeros(image.shape[:2],np.float32)
    monkeypatch.setattr(ecgdig.inkmodel,'predict',predict)
    classic=digitize(sample)
    hybrid=digitize(sample,Config(ink_model='fake',ink_mix=0))
    np.testing.assert_array_equal(classic['signals'],hybrid['signals'])
    assert classic['layout_rows']==hybrid['layout_rows']
    assert seen==['layout','layout','neural']
