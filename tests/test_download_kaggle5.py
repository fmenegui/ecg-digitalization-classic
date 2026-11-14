"""Exercise the full-download fallback without a network or large checkpoint."""
from collections import OrderedDict
import json
import shutil
import pytest

@pytest.mark.parametrize('mode',['auto','full'])
def test_full_checkpoint_fallback(tmp_path,monkeypatch,mode):
    torch=pytest.importorskip('torch')
    from hybrid import download_kaggle5 as downloader
    expected=OrderedDict(weight=torch.arange(12,dtype=torch.float32).reshape(3,4),counter=torch.tensor(5))
    source=tmp_path/'fixture.ckpt'
    torch.save({'ema_state_dict':{0.999:OrderedDict(('model.'+k,v) for k,v in expected.items())}},source)
    class NoRanges:
        def __init__(self,url): raise RuntimeError('ranges disabled by test server')
    monkeypatch.setattr(downloader,'Remote',NoRanges)
    def fetch(url,target,timeout=900):
        if url.endswith(downloader.FILE): shutil.copyfile(source,target)
        else: target.write_text('model:\n  _target_: ecg.model.signal_fine_heatmap_model.SignalFineHeatmapModel\n')
    monkeypatch.setattr(downloader,'curl_file',fetch)
    out=tmp_path/'out'
    meta=downloader.download(out,mode)
    actual=torch.load(out/'ema.pt',weights_only=True)
    for key,value in expected.items(): torch.testing.assert_close(actual[key],value)
    assert meta['download_mode']=='full'
    assert meta['tensors']==2
    assert len(meta['sha256'])==64
    assert json.loads((out/'provenance.json').read_text())==meta
    assert not (out/'source.ckpt').exists()

def test_failed_download_does_not_mark_success(tmp_path,monkeypatch):
    from hybrid import download_kaggle5 as downloader
    def fail(*args,**kwargs): raise RuntimeError('HTTP 403 from test server')
    monkeypatch.setattr(downloader,'curl_file',fail)
    with pytest.raises(RuntimeError,match='HTTP 403'):
        downloader.download(tmp_path,'full')
    assert not (tmp_path/'provenance.json').exists()
