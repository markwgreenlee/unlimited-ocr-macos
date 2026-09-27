#!/usr/bin/env python3
"""
Unlimited-OCR (baidu/Unlimited-OCR) on Apple Silicon.

The upstream model code is written for NVIDIA GPUs: it calls `.cuda()` on every
input tensor and wraps generation in `torch.autocast("cuda", ...)`. This script
installs a small runtime shim that redirects those calls to MPS (or CPU) without
modifying the downloaded model files, then exposes a simple CLI.

Usage
-----
  ./run-ocr.sh page.png                      # single image, gundam mode
  ./run-ocr.sh scan.pdf --out out/scan        # PDF -> multi-page parsing
  ./run-ocr.sh page.png --mode base           # single-shot 1024px, no tiling
  ./run-ocr.sh fig.png --prompt "<image>Parse the figure."
  ./run-ocr.sh page.png --device cpu          # force CPU

Output goes to <--out>/result.md (plus result_with_boxes.jpg and images/ crops).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = os.environ.get("UNLIMITED_OCR_MODEL", str(ROOT / "model"))

# Let ops MPS has not implemented fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Bound before any patching: the workaround below installs itself as
# Tensor.masked_scatter_, so it must call the original implementation, not the
# class attribute (which by then is itself).
_MASKED_SCATTER_ = torch.Tensor.masked_scatter_


def masked_scatter_via_cpu(target: torch.Tensor, mask: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """`Tensor.masked_scatter_` with the write performed on CPU.

    On MPS (torch 2.10, macOS 27) `masked_scatter_` with a broadcast mask — the
    model writes image features in with a (seq, 1) mask against a (seq, hidden)
    embedding block — silently writes nothing, so the decoder never sees the
    image and immediately emits EOS. The round trip is one small tensor and
    `copy_` writes back through views, so the caller's semantics are unchanged.
    """
    staged = target.detach().to("cpu")
    _MASKED_SCATTER_(staged, mask.detach().to("cpu"), source.detach().to("cpu"))
    target.copy_(staged)
    return target


def install_cuda_shim(device: str, dtype: torch.dtype = torch.bfloat16) -> None:
    """Redirect the model code's hard-coded CUDA calls to `device`.

    The model code also hard-codes `.to(torch.bfloat16)` on the transformed image
    tensors, which crashes when the weights are loaded in another dtype
    ("Input type (c10::BFloat16) and bias type (c10::Half) should be the same"),
    so bf16 casts are redirected to the dtype actually in use.
    """
    target = torch.device(device)

    torch.Tensor.cuda = lambda self, *a, **k: self.to(target)  # type: ignore[method-assign]
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(target)  # type: ignore[method-assign]

    if device == "mps":
        torch.Tensor.masked_scatter_ = masked_scatter_via_cpu  # type: ignore[method-assign]

    if dtype is not torch.bfloat16:
        _to = torch.Tensor.to

        def to(self, *a, **k):
            a = tuple(dtype if x is torch.bfloat16 else x for x in a)
            if k.get("dtype") is torch.bfloat16:
                k = {**k, "dtype": dtype}
            return _to(self, *a, **k)

        torch.Tensor.to = to  # type: ignore[method-assign]

    _autocast = torch.autocast

    def autocast(device_type="cuda", **kwargs):
        # Weights and image tensors are already in a single dtype, so autocast is
        # a no-op for inference; on non-CUDA devices just neutralise it.
        if device_type == "cuda" and not torch.cuda.is_available():
            return contextlib.nullcontext()
        return _autocast(device_type, **kwargs)

    torch.autocast = autocast  # type: ignore[assignment]


def load_model(model_dir: str, device: str, dtype: torch.dtype):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=dtype,
        # The language model's ATTENTION_CLASSES has no "sdpa" entry; only
        # "eager" (and flash-attention, which needs CUDA) are wired up.
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model = model.eval().to(device)
    return tokenizer, model


def pdf_to_images(pdf_path: str, dpi: int, out_dir: Path) -> list[str]:
    import fitz  # PyMuPDF

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = out_dir / f"page_{i + 1:04d}.png"
        page.get_pixmap(matrix=mat).save(str(out))
        paths.append(str(out))
    doc.close()
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description="Unlimited-OCR on Apple Silicon")
    p.add_argument("input", help="image file, PDF file, or directory of images")
    p.add_argument("--out", default=None, help="output directory (default: ./out/<input stem>)")
    p.add_argument("--mode", choices=["gundam", "base"], default="gundam",
                   help="gundam = 1024 base + 640 tiles (default); base = single 1024px view")
    p.add_argument("--prompt", default=None,
                   help="default: '<image>document parsing.' (single) / '<image>Multi page parsing.'")
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--dpi", type=int, default=300, help="PDF rasterisation DPI (default 300)")
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--model", default=DEFAULT_MODEL_DIR)
    p.add_argument("--quiet", action="store_true", help="do not stream tokens to stdout")
    args = p.parse_args()

    src = Path(args.input).expanduser()
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1

    out_dir = Path(args.out).expanduser() if args.out else ROOT / "out" / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    dtype = DTYPES[args.dtype]
    if device == "cpu" and args.dtype == "fp16":
        print("note: fp16 is slow on CPU; bf16 or fp32 is usually better", file=sys.stderr)

    print(f"device={device} dtype={args.dtype} model={args.model}")
    install_cuda_shim(device, dtype)

    t0 = time.time()
    tokenizer, model = load_model(args.model, device, dtype)
    print(f"model loaded in {time.time() - t0:.0f}s")

    # Decide single-image vs multi-page.
    if src.is_dir():
        pages = sorted(str(f) for f in src.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if not pages:
            print(f"error: no images in {src}", file=sys.stderr)
            return 1
    elif src.suffix.lower() == ".pdf":
        pages = pdf_to_images(str(src), args.dpi, out_dir / "pages")
        print(f"rasterised {len(pages)} page(s) at {args.dpi} dpi")
    else:
        pages = [str(src)]

    t0 = time.time()
    if len(pages) == 1:
        prompt = args.prompt or "<image>document parsing."
        base_size, image_size, crop_mode = (1024, 640, True) if args.mode == "gundam" else (1024, 1024, False)
        # eval_mode=True returns the text and skips the model's own saving/streaming,
        # so in that case we write result.md ourselves.
        text = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=pages[0],
            output_path=str(out_dir),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            max_length=args.max_length,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=not args.quiet,
            eval_mode=bool(args.quiet),
        )
        if args.quiet and isinstance(text, str):
            import re
            # eval_mode skips upstream's post-processing, so drop the grounding tags.
            clean = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text).replace("<|ref|>", "").replace("<|/ref|>", "")
            (out_dir / "result.md").write_text(clean.strip() + "\n", encoding="utf-8")
    else:
        # Multi-page parsing only supports the 1024px base view.
        prompt = args.prompt or "<image>Multi page parsing."
        model.infer_multi(
            tokenizer,
            prompt=prompt,
            image_files=pages,
            output_path=str(out_dir),
            image_size=1024,
            max_length=args.max_length,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )
    dt = time.time() - t0

    result = out_dir / "result.md"
    if result.exists():
        print(f"\ndone in {dt:.0f}s -> {result} ({result.stat().st_size} bytes)")
    else:
        print(f"\ndone in {dt:.0f}s, but {result} was not written", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
