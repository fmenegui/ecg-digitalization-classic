"""Paired development pilot, not a hidden test or clinical validation.

Run from the repository root:
    python -m benchmarks.wsp_pilot --out results/wsp_pilot --cases 3

Known layout, duration, speed and gain isolate the segmentation intervention.
Geometry is estimated by the same classical method in both arms. Missing signal
samples are zero-filled for the error metric and separately counted in coverage.
Failure cases remain in the metrics with an entirely missing prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import scipy
import torch

from ecgdig.digitize import Config, digitize
from ecgdig.programmed_unet import ProgrammedSupport, read_spec
from ecgdig.synth import synth_signals, render


def signal_metrics(truth, prediction, shown):
    finite = np.isfinite(prediction)
    predicted = np.nan_to_num(prediction, nan=0, posinf=0, neginf=0)
    gt, pr = truth[shown].astype(float), predicted[shown].astype(float)
    error = np.sum((gt - pr) ** 2)
    power = np.sum(gt ** 2)
    return dict(snr_db=float(10 * np.log10(max(power, 1e-20) / max(error, 1e-20))),
                rmse_mv=float(np.sqrt(np.mean((gt - pr) ** 2))),
                coverage=float(finite[shown].mean()),
                missing_samples=int((~finite & shown).sum()),
                displayed_samples=int(shown.sum()))


def run(out, cases=3, seed=91000, threads=2, spec=None):
    if cases < 1 or threads < 1:
        raise ValueError("cases and threads must be positive")
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    helper = ProgrammedSupport(spec)
    cfg = Config(R=5, speed=25, gain=10, duration=10, layout="3x4+1")
    report = dict(kind="synthetic_development_pilot", hidden_test=False,
                  known_metadata=["layout", "speed", "gain", "duration"],
                  geometry="classical automatic", model=helper.model.audit(),
                  metric="aggregate displayed-sample SNR, no alignment, missing=zero",
                  official_challenge_metric=False, cases=[])
    report["software"] = dict(python=platform.python_version(), torch=torch.__version__,
                              numpy=np.__version__, scipy=scipy.__version__,
                              opencv=cv2.__version__, device="cpu", threads=threads)
    levels = ["none", "medium", "heavy"]
    for i in range(cases):
        rng = np.random.default_rng(seed + i)
        truth = synth_signals(rng)
        level = levels[i % len(levels)]
        image, shown, meta = render(truth, rng, ppmm=5.5, degrade=level)
        cv2.imwrite(str(out / f"case_{i:02d}_input.png"), image)
        entry = dict(seed=seed + i, degradation=level)
        for method, intervention in (("classic", None), ("programmed_unet", helper)):
            start = time.perf_counter()
            try:
                result = digitize(image, cfg, return_debug=True, trace_support=intervention)
                prediction = result["signals"]
                if prediction.shape != truth.shape:
                    raise ValueError(f"Unexpected signal shape {prediction.shape}")
                status = dict(status="ok", layout=result["layout"],
                              support=result.get("trace_support"))
                cv2.imwrite(str(out / f"case_{i:02d}_{method}.png"), result["debug_image"])
            except Exception as exc:
                prediction = np.full_like(truth, np.nan)
                status = dict(status="failed", error=f"{type(exc).__name__}: {exc}")
            status.update(signal_metrics(truth, prediction, shown))
            status["seconds"] = time.perf_counter() - start
            entry[method] = status
        entry["delta_snr_db"] = entry["programmed_unet"]["snr_db"] - entry["classic"]["snr_db"]
        report["cases"].append(entry)
        print(json.dumps(dict(case=i, degradation=level, delta_snr_db=entry["delta_snr_db"],
                              classic=entry["classic"]["snr_db"],
                              programmed_unet=entry["programmed_unet"]["snr_db"])), flush=True)
        (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    report["summary"] = {
        method: dict(mean_snr_db=float(np.mean([r[method]["snr_db"] for r in report["cases"]])),
                     mean_rmse_mv=float(np.mean([r[method]["rmse_mv"] for r in report["cases"]])),
                     mean_coverage=float(np.mean([r[method]["coverage"] for r in report["cases"]])),
                     failures=sum(r[method]["status"] != "ok" for r in report["cases"]),
                     mean_seconds=float(np.mean([r[method]["seconds"] for r in report["cases"]])))
        for method in ("classic", "programmed_unet")
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/wsp_pilot")
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--seed", type=int, default=91000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--recipe", help="JSON recipe or exported model audit")
    args = ap.parse_args(argv)
    run(args.out, args.cases, args.seed, args.threads, read_spec(args.recipe) if args.recipe else None)


if __name__ == "__main__":
    main()
