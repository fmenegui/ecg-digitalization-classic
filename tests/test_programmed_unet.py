"""Verify the programmed computation and the boundary with classical fitting."""
import importlib
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from ecgdig.programmed_unet import (
    ProgrammedTraceUNet, ProgramSpec, ProgrammedSupport,
    predict_features, assist_mask, make_features,
    read_spec,
)


def test_weights_are_a_complete_deterministic_frozen_program():
    torch.manual_seed(1)
    a = ProgrammedTraceUNet()
    torch.manual_seed(29)
    b = ProgrammedTraceUNet()
    assert a.audit() == b.audit()
    assert all(not p.requires_grad for p in a.parameters())
    assert a.audit()["nonzero_parameters"] < a.audit()["parameters"]
    assert a.audit()["gradient_steps"] == 0


def test_constant_field_matches_specification():
    model = ProgrammedTraceUNet()
    x = torch.zeros(2, 3, 31, 39)
    x[:, 0] = 0.2
    x[:, 1] = 0.1
    x[:, 2] = 1
    # Directional and local contrast channels are zero on a constant field.
    expected_logit = 6 * 0.2 + 0.5 * 0.1 - 2
    np.testing.assert_allclose(model(x).numpy(), expected_logit, atol=2e-6)


def test_saved_checkpoint_and_audit_rebuild_identical_weights(tmp_path):
    import json
    model = ProgrammedTraceUNet()
    path = tmp_path / "weights.pt"
    audit_path = tmp_path / "audit.json"
    torch.save({"state_dict": model.state_dict(), "audit": model.audit()}, path)
    audit_path.write_text(json.dumps(model.audit()))
    rebuilt = ProgrammedTraceUNet(read_spec(audit_path))
    checkpoint = torch.load(path, weights_only=True)
    rebuilt.load_state_dict(checkpoint["state_dict"])
    assert rebuilt.audit() == model.audit()


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (67, 91), (137, 193)])
def test_tiling_preserves_full_computation_at_odd_shapes(shape):
    x = np.random.default_rng(140).random((3, *shape), dtype=np.float32)
    model = ProgrammedTraceUNet()
    full = predict_features(model, x, tile_size=0)
    tiled = predict_features(model, x, tile_size=64)
    assert full.shape == shape
    np.testing.assert_allclose(tiled, full, atol=3e-6)


def test_coarse_encoder_and_fine_skip_both_affect_output():
    x = torch.zeros(1, 3, 64, 64)
    x[:, 0, 31, 8:56] = 0.8
    default = ProgrammedTraceUNet()(x)
    fine_only = ProgrammedTraceUNet(ProgramSpec(decoder_context=0))(x)
    coarse_only = ProgrammedTraceUNet(ProgramSpec(decoder_context=1))(x)
    assert not torch.allclose(default, fine_only)
    assert not torch.allclose(default, coarse_only)
    a = ProgrammedTraceUNet(ProgramSpec(coarse_context=0))(x)
    b = ProgrammedTraceUNet(ProgramSpec(coarse_context=1))(x)
    assert not torch.allclose(a, b)


def test_no_evidence_is_not_converted_to_ink():
    ink = np.zeros((21, 21), np.float32)
    mask = np.zeros_like(ink, bool)
    mask[10, 10] = True
    # Even maximum scores cannot invent optical evidence in empty pixels.
    result = assist_mask(ink, mask, np.ones_like(ink), R=5)
    np.testing.assert_array_equal(result, mask)
    ink[10, 11] = 0.3
    ink[1, 1] = 0.3
    result = assist_mask(ink, mask, np.ones_like(ink), R=5)
    assert result[10, 11] and not result[1, 1]
    assert result[mask].all()


def test_white_paper_and_feature_validation():
    rect = np.full((71, 85, 3), 255, np.uint8)
    ink = np.zeros(rect.shape[:2], np.float32)
    paper = np.ones_like(ink, np.uint8)
    f = make_features(rect, paper, 5, ink)
    scores = predict_features(ProgrammedTraceUNet(), f, tile_size=64)
    assert scores.max() < 0.2
    with pytest.raises(ValueError):
        make_features(rect.astype(float), paper, 5, ink)
    with pytest.raises(ValueError):
        predict_features(ProgrammedTraceUNet(), f, tile_size=63)
    with pytest.raises(ValueError):
        ProgramSpec(decoder_context=float("nan"))


def test_classical_import_has_no_torch_dependency():
    code = '''
import sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is deliberately unavailable")
sys.meta_path.insert(0, NoTorch())
import ecgdig
assert "torch" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_identity_hook_preserves_end_to_end_signal():
    import cv2
    from ecgdig.synth import render, synth_signals
    from ecgdig.digitize import Config, digitize
    cv2.setNumThreads(2)
    rng = np.random.default_rng(24017)
    img, _, _ = render(synth_signals(rng), rng, ppmm=5, degrade="none")
    cfg = Config(R=5, speed=25, gain=10, duration=10, layout="3x4+1")
    expected = digitize(img, cfg)
    actual = digitize(img, cfg, trace_support=lambda r, p, R, d, b: (b, {"test": True}))
    np.testing.assert_array_equal(actual["signals"], expected["signals"])
    assert actual["trace_support"] == {"test": True}


def test_hook_rejects_wrong_coordinate_system(monkeypatch):
    module = importlib.import_module("ecgdig.digitize")
    image = np.full((80, 80, 3), 255, np.uint8)
    monkeypatch.setattr(module, "rectify", lambda *a, **k: (image, np.ones((80, 80), np.uint8), {}))
    monkeypatch.setattr(module, "ink_map", lambda *a, **k: (np.zeros((80, 80), np.float32), 0))
    monkeypatch.setattr(module, "binarize", lambda *a, **k: np.zeros((80, 80), bool))
    with pytest.raises(ValueError, match="matching boolean"):
        module.digitize(image, trace_support=lambda *a: (np.zeros((79, 80), bool), {}))
