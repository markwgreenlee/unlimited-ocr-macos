#!/usr/bin/env python3
"""
Diagnose empty / garbage output from Unlimited-OCR on MPS.

Runs the model on a small test page for each requested device+dtype, with a
forward hook on every submodule, and reports:
  * the first modules (in execution order) whose output contains NaN/Inf
  * the decoded text, so an empty generation is visible

Usage:  ./.venv/bin/python diagnose-mps.py
        ./.venv/bin/python diagnose-mps.py --configs mps:bf16,mps:fp16,cpu:bf16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # PYTHONSAFEPATH may suppress the script-dir entry

import ocr  # reuses the CUDA->MPS shim and loader from ocr.py

import torch


def nonfinite_names(model, limit: int = 8):
    hits: list[str] = []
    handles = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            tensors = out if isinstance(out, (tuple, list)) else (out,)
            for t in tensors:
                if torch.is_tensor(t) and t.is_floating_point():
                    if not torch.isfinite(t).all():
                        if name not in hits and len(hits) < limit:
                            hits.append(name)
                        return
        return hook

    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(make_hook(name)))
    return hits, handles


def run(device: str, dtype_name: str, image: str, max_length: int) -> None:
    dtype = ocr.DTYPES[dtype_name]
    print("=" * 70)
    print(f"device={device} dtype={dtype_name}")
    ocr.install_cuda_shim(device, dtype)
    tokenizer, model = ocr.load_model(ocr.DEFAULT_MODEL_DIR, device, dtype)

    hits, handles = nonfinite_names(model)
    out_dir = ROOT / "out" / f"diag_{device}_{dtype_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        text = model.infer(
            tokenizer,
            prompt="<image>document parsing.",
            image_file=image,
            output_path=str(out_dir),
            base_size=1024, image_size=1024, crop_mode=False,
            max_length=max_length,
            no_repeat_ngram_size=35, ngram_window=128,
            save_results=False, eval_mode=True,
        )
    finally:
        for h in handles:
            h.remove()

    text = text if isinstance(text, str) else ""
    print(f"  decoded chars : {len(text)}")
    print(f"  decoded text  : {text[:200]!r}")
    print(f"  non-finite in : {hits if hits else 'none — all module outputs finite'}")
    del model
    if device == "mps":
        torch.mps.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="mps:bf16,mps:fp16",
                   help="comma-separated device:dtype pairs (default mps:bf16,mps:fp16)")
    p.add_argument("--image", default=str(ROOT / "samples" / "synthetic_page.png"))
    p.add_argument("--max-length", type=int, default=1024)
    args = p.parse_args()

    print(f"torch {torch.__version__} | mps available: {torch.backends.mps.is_available()}")
    for cfg in args.configs.split(","):
        device, _, dtype_name = cfg.partition(":")
        try:
            run(device.strip(), (dtype_name or "bf16").strip(), args.image, args.max_length)
        except Exception as exc:  # keep going through the remaining configs
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
