"""An experimental U-Net whose weights encode explicit image-processing rules.

No data, optimizer, pretrained checkpoint or gradients are used to build the
model. This is a deliberately small, constrained U-Net, not a trained medical
image segmenter. Sigmoid outputs are segmentation scores, not calibrated
probabilities. The classical optical density is kept for subpixel fitting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("Programmed U-Net requires the optional extra: pip install '.[wsp]'") from exc


@dataclass(frozen=True)
class ProgramSpec:
    """The complete recipe, with no fitted constants or data-dependent weights."""

    decoder_context: float = 0.25
    coarse_context: float = 0.5
    ink_weight: float = 6.0
    neutral_weight: float = 0.5
    ridge_weight: float = 0.75
    contrast_weight: float = 0.5
    bias: float = -2.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("All recipe values must be finite")
        if not 0 <= self.decoder_context <= 1 or not 0 <= self.coarse_context <= 1:
            raise ValueError("Context mixtures must be in [0, 1]")


FEATURE_NAMES = (
    "classical_ink", "neutral_darkness", "horizontal_ridge", "vertical_ridge",
    "rising_ridge", "falling_ridge", "local_contrast", "paper",
)


def read_spec(path):
    """Read either a plain recipe or the JSON audit exported with a checkpoint."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Recipe JSON must be an object")
    return ProgramSpec(**data.get("recipe", data))


class ProgrammedTraceUNet(nn.Module):
    """3 -> 8 -> 8 -> 8 -> 8 -> 8 -> 1, two pools and two skip connections.

    The encoder separates ink, color-neutral darkness, four directional ridges
    and local contrast. Pooled features provide context at 1/2 and 1/4 scale.
    The decoder mixes that context with high-resolution skip features. A 1x1
    head combines the evidence into logits. Every tensor is written explicitly.
    """

    def __init__(self, spec: ProgramSpec | None = None):
        super().__init__()
        self.spec = spec or ProgramSpec()
        self.enc0 = nn.Conv2d(3, 8, 5, padding=2, padding_mode="replicate")
        self.enc1 = nn.Conv2d(8, 8, 3, padding=1, padding_mode="replicate")
        self.bottleneck = nn.Conv2d(8, 8, 3, padding=1, padding_mode="replicate")
        self.dec1 = nn.Conv2d(16, 8, 1)
        self.dec0 = nn.Conv2d(16, 8, 1)
        self.head = nn.Conv2d(8, 1, 1)
        self.compile_weights()

    @torch.no_grad()
    def compile_weights(self):
        for p in self.parameters():
            p.zero_()
        w = self.enc0.weight
        w[0, 0, 2, 2] = 1
        w[1, 1, 2, 2] = 1
        # Center-minus-flanks at four angles, applied to classical ink.
        for channel, (dy, dx) in enumerate(((2, 0), (0, 2), (2, 2), (2, -2)), 2):
            w[channel, 0, 2, 2] = 1
            w[channel, 0, 2 + dy, 2 + dx] = -0.5
            w[channel, 0, 2 - dy, 2 - dx] = -0.5
        w[6, 0, :, :] = -1 / 25
        w[6, 0, 2, 2] += 1
        w[7, 2, 2, 2] = 1
        for layer in (self.enc1, self.bottleneck):
            for c in range(8):
                layer.weight[c, c, 1, 1] = 1
        for layer, mixture in ((self.dec1, self.spec.coarse_context),
                               (self.dec0, self.spec.decoder_context)):
            for c in range(8):
                layer.weight[c, c, 0, 0] = 1 - mixture
                layer.weight[c, c + 8, 0, 0] = mixture
        self.head.weight[0, 0, 0, 0] = self.spec.ink_weight
        self.head.weight[0, 1, 0, 0] = self.spec.neutral_weight
        self.head.weight[0, 2:6, 0, 0] = self.spec.ridge_weight
        self.head.weight[0, 6, 0, 0] = self.spec.contrast_weight
        self.head.bias[0] = self.spec.bias
        self.requires_grad_(False)
        return self

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected a nonempty Bx3xHxW tensor")
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        e0 = F.relu(self.enc0(x))
        e1 = F.relu(self.enc1(F.avg_pool2d(e0, 2)))
        mid = F.relu(self.bottleneck(F.avg_pool2d(e1, 2)))
        d1 = F.relu(self.dec1(torch.cat((e1, F.interpolate(mid, size=e1.shape[-2:], mode="nearest")), 1)))
        d0 = F.relu(self.dec0(torch.cat((e0, F.interpolate(d1, size=e0.shape[-2:], mode="nearest")), 1)))
        return self.head(d0)[..., :h, :w]

    def audit(self):
        digest = hashlib.sha256()
        nonzero = 0
        total = 0
        for name, tensor in self.state_dict().items():
            array = tensor.detach().cpu().numpy().astype("<f4")
            digest.update(name.encode())
            digest.update(array.tobytes())
            nonzero += int(np.count_nonzero(array))
            total += array.size
        return dict(architecture="ProgrammedTraceUNet-v1", parameters=total,
                    nonzero_parameters=nonzero, weights_sha256=digest.hexdigest(),
                    gradient_steps=0, pretrained_weights=False,
                    recipe=asdict(self.spec), channels=list(FEATURE_NAMES),
                    scores_calibrated=False)


def make_features(rect, paper, R, ink):
    """Inputs are a rectified BGR uint8 image, paper mask, px/mm and ink map."""
    from .extract import _bg
    rect, paper, ink = np.asarray(rect), np.asarray(paper), np.asarray(ink)
    if rect.ndim != 3 or rect.shape[2] != 3 or rect.dtype != np.uint8:
        raise ValueError("rect must be an HxWx3 BGR uint8 image")
    if ink.shape != rect.shape[:2] or paper.shape != ink.shape or ink.size == 0:
        raise ValueError("Image, paper and ink dimensions must match and be nonempty")
    if not np.isfinite(R) or R <= 0 or not np.isfinite(ink).all():
        raise ValueError("R must be positive and ink must be finite")
    neutral = rect.max(axis=-1).astype(np.float32)
    neutral = 1 - np.clip(neutral / (_bg(neutral, R) + 1e-3), 0, 1)
    valid = (paper > 0).astype(np.float32)
    return np.stack((np.clip(ink, 0, 1) * valid, neutral * valid, valid)).astype(np.float32)


@torch.inference_mode()
def predict_features(model, features, tile_size=384, device="cpu"):
    """Tile with a halo and aligned pool phases, preserving full-image output."""
    f = np.asarray(features, np.float32)
    if f.ndim != 3 or f.shape[0] != 3 or min(f.shape[1:]) < 1 or not np.isfinite(f).all():
        raise ValueError("features must be a finite, nonempty 3xHxW array")
    if tile_size and (tile_size < 32 or tile_size % 4):
        raise ValueError("tile_size must be 0 or a multiple of four >= 32")
    model = model.to(device).eval()
    h, w = f.shape[1:]
    if not tile_size:
        return model(torch.from_numpy(f[None]).to(device)).sigmoid()[0, 0].cpu().numpy()
    out = np.empty((h, w), np.float32)
    halo = 32
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1, x1 = min(h, y + tile_size), min(w, x + tile_size)
            ya, xa = max(0, y - halo), max(0, x - halo)
            yb, xb = min(h, y1 + halo), min(w, x1 + halo)
            tile = np.ascontiguousarray(f[:, ya:yb, xa:xb])
            p = model(torch.from_numpy(tile[None]).to(device)).sigmoid()[0, 0].cpu().numpy()
            out[y:y1, x:x1] = p[y - ya:y1 - ya, x - xa:x1 - xa]
    return out


def assist_mask(ink, classical_mask, scores, R, threshold=0.55, radius_mm=0.3, ink_floor=0.1):
    """Add only optical evidence close to existing classical support.

    This cannot repair absent ink, separate overlapping leads or infer layout.
    It deliberately preserves the measured optical density and all old support.
    """
    import cv2
    ink, classical_mask, scores = np.asarray(ink), np.asarray(classical_mask, bool), np.asarray(scores)
    if ink.shape != classical_mask.shape or ink.shape != scores.shape or ink.ndim != 2:
        raise ValueError("Ink, mask and scores must be matching 2D arrays")
    if not (np.isfinite(ink).all() and np.isfinite(scores).all()):
        raise ValueError("Ink and scores must be finite")
    if not 0 < threshold < 1 or not 0 <= ink_floor <= 1:
        raise ValueError("Invalid support threshold")
    if not np.isfinite(R) or R <= 0 or not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError("R must be positive and radius_mm nonnegative")
    radius = int(np.ceil(radius_mm * R))
    near = cv2.dilate(classical_mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0
    additions = near & (scores >= threshold) & (ink >= ink_floor)
    return classical_mask | additions


class ProgrammedSupport:
    """Callable hook for digitize(..., trace_support=ProgrammedSupport())."""

    def __init__(self, spec=None, threshold=0.55, radius_mm=0.3, ink_floor=0.1,
                 tile_size=384, device="cpu"):
        self.model = ProgrammedTraceUNet(spec)
        self.threshold, self.radius_mm, self.ink_floor = threshold, radius_mm, ink_floor
        self.tile_size, self.device = tile_size, device

    def __call__(self, rect, paper, R, ink, classical_mask):
        scores = predict_features(self.model, make_features(rect, paper, R, ink), self.tile_size, self.device)
        mask = assist_mask(ink, classical_mask, scores, R, self.threshold, self.radius_mm, self.ink_floor)
        info = dict(self.model.audit(), threshold=self.threshold, radius_mm=self.radius_mm,
                    ink_floor=self.ink_floor, added_pixels=int(np.count_nonzero(mask & ~classical_mask)),
                    classical_pixels=int(np.count_nonzero(classical_mask)), optical_density_preserved=True)
        return mask, info


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="programmed_unet.pt")
    args = parser.parse_args(argv)
    model = ProgrammedTraceUNet()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), audit=model.audit()), path)
    path.with_suffix(".json").write_text(json.dumps(model.audit(), indent=2) + "\n")
    print(json.dumps(model.audit(), indent=2))


if __name__ == "__main__":
    main()
