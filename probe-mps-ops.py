#!/usr/bin/env python3
"""
Op-level MPS vs CPU comparison for the primitives Unlimited-OCR depends on.

All module outputs were finite on MPS yet generation emitted zero tokens, so the
failure is a wrong-but-finite result somewhere. This runs each suspect op on
identical inputs on CPU and on MPS and reports the disagreement. No model load,
so it takes seconds.

Usage:  ./.venv/bin/python probe-mps-ops.py
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

DT = torch.bfloat16
ROWS: list[tuple[str, str]] = []


def report(name: str, verdict: str) -> None:
    ROWS.append((name, verdict))
    print(f"  {name:38s} {verdict}")


def cmp_tensor(name: str, cpu_out: torch.Tensor, mps_out: torch.Tensor) -> None:
    a = cpu_out.float().cpu()
    b = mps_out.float().cpu()
    if a.shape != b.shape:
        report(name, f"SHAPE MISMATCH cpu={tuple(a.shape)} mps={tuple(b.shape)}")
        return
    diff = (a - b).abs().max().item()
    nonfinite = not torch.isfinite(b).all().item()
    tag = "MISMATCH" if diff > 0.05 else "ok"
    extra = " NON-FINITE" if nonfinite else ""
    report(name, f"{tag:9s} max|cpu-dev|={diff:.4g}{extra}")


def cmp_index(name: str, cpu_idx: torch.Tensor, mps_idx: torch.Tensor) -> None:
    a = cpu_idx.cpu()
    b = mps_idx.cpu()
    agree = (a == b).float().mean().item() * 100
    tag = "ok" if agree == 100.0 else "MISMATCH"
    report(name, f"{tag:9s} index agreement={agree:.1f}%")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", help="device to compare against CPU (default mps)")
    DEV = ap.parse_args().device
    print(f"torch {torch.__version__} | mps available: {torch.backends.mps.is_available()} | comparing cpu vs {DEV}")
    if DEV == "mps" and not torch.backends.mps.is_available():
        print("MPS not available here — nothing to compare.")
        return 1
    torch.manual_seed(0)
    print("\nop comparisons (bfloat16 unless noted):")

    # 1. embedding lookup — how input_ids become inputs_embeds
    w = torch.randn(1000, 64, dtype=DT)
    ids = torch.randint(0, 1000, (37,))
    cmp_tensor("embedding lookup", w[ids], w.to(DEV)[ids.to(DEV)])

    # 2. masked_scatter_ — how image features are written into inputs_embeds.
    #    The model does inputs_embeds[idx].masked_scatter_(mask, feats), i.e. an
    #    in-place write through a view of a 3-D tensor.
    def masked_scatter(dev, how: str):
        base = torch.zeros(1, 10, 4, dtype=DT, device=dev)
        mask = torch.zeros(10, 1, dtype=torch.bool, device=dev)
        mask[2] = mask[5] = mask[7] = True
        src = torch.arange(1, 13, dtype=DT, device=dev).reshape(3, 4)
        if how == "view":
            base[0].masked_scatter_(mask, src)      # the model's pattern
        elif how == "contiguous":
            b2 = base[0].clone()
            b2.masked_scatter_(mask, src)
            base[0] = b2
        elif how == "patched":
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import ocr
            ocr.masked_scatter_via_cpu(base[0], mask, src)   # the shim's workaround
        return base

    ref = masked_scatter("cpu", "view")
    cmp_tensor("masked_scatter_ (contiguous)", ref, masked_scatter(DEV, "contiguous"))
    cmp_tensor("masked_scatter_ (via view)  ", ref, masked_scatter(DEV, "view"))
    cmp_tensor("masked_scatter_ (patched)   ", ref, masked_scatter(DEV, "patched"))

    # 3. greedy decode: argmax over the vocabulary
    logits = torch.randn(1, 129280, dtype=DT)
    cmp_index("argmax over vocab", logits.argmax(-1), logits.to(DEV).argmax(-1))

    # 4. MoE router: top-6 of 64 experts
    gate = torch.randn(37, 64, dtype=DT)
    cmp_index("topk(6) expert routing", gate.topk(6, -1).indices, gate.to(DEV).topk(6, -1).indices)

    # 5. attention softmax, computed in fp32 then cast back (model's pattern)
    att = torch.randn(1, 10, 37, 37, dtype=DT)
    f = lambda t: torch.softmax(t, dim=-1, dtype=torch.float32).to(DT)
    cmp_tensor("softmax(fp32)->bf16", f(att), f(att.to(DEV)))

    # 6. lm_head projection: bf16 matmul with a large output dim
    h = torch.randn(1, 1280, dtype=DT)
    head = torch.randn(1280, 129280, dtype=DT)
    cmp_tensor("lm_head matmul (1280x129280)", h @ head, h.to(DEV) @ head.to(DEV))

    # 7. RMSNorm pattern
    x = torch.randn(4, 1280, dtype=DT)
    rms = lambda t: (t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(DT))
    cmp_tensor("rmsnorm", rms(x), rms(x.to(DEV)))

    # 8. SAM patch embedding: strided conv2d
    img = torch.randn(1, 3, 128, 128, dtype=DT)
    conv = torch.nn.Conv2d(3, 32, kernel_size=16, stride=16, dtype=DT)
    cmp_tensor("conv2d 16x16 stride16", conv(img), conv.to(DEV)(img.to(DEV)))

    # 9. bilinear interpolation of position embeddings
    pe = torch.randn(1, 8, 16, 16, dtype=DT)
    itp = lambda t: torch.nn.functional.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)
    cmp_tensor("interpolate bilinear", itp(pe), itp(pe.to(DEV)))

    # 10. cumsum (position bookkeeping) in bf16
    c = torch.randn(1, 64, dtype=DT)
    cmp_tensor("cumsum", c.cumsum(-1), c.to(DEV).cumsum(-1))

    # 11. the workaround as actually installed: patch the class attribute, then
    #     reach the op through the method. This is the configuration that once
    #     recursed 988 frames deep (the helper called the patched attribute
    #     instead of the original), and it runs on CPU too, so the --device cpu
    #     self-test covers it.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ocr

    torch.Tensor.masked_scatter_ = ocr.masked_scatter_via_cpu
    try:
        cmp_tensor("masked_scatter_ (shim installed)", ref, masked_scatter(DEV, "view"))
    except RecursionError:
        report("masked_scatter_ (shim installed)", "RECURSION — helper is calling itself")
    finally:
        torch.Tensor.masked_scatter_ = ocr._MASKED_SCATTER_

    bad = [n for n, v in ROWS if not v.startswith("ok")]
    print("\nsummary:", "ALL OPS AGREE" if not bad else f"{len(bad)} disagreement(s): " + ", ".join(b.strip() for b in bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
