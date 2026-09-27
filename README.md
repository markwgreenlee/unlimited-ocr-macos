# Unlimited-OCR on Apple Silicon

Run [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) — a DeepSeek-OCR-derived
vision-language model for one-shot document parsing — locally on an M-series Mac, on the GPU
via Metal (MPS). Ships an installer, a CLI, correctness diagnostics, and a
[Claude Science](https://claude.ai) skill so an agent can drive it.

Upstream supports NVIDIA only: all three documented paths (HF `transformers` with `.cuda()`,
vLLM Docker images, SGLang with `--attention-backend fa3`) require CUDA. This repo is the
Apple Silicon port — plus the fix for a silent Metal bug that otherwise makes the model
return **nothing at all** while appearing to succeed.

## What you get

- **A 6.7 GB local OCR model.** Page images → Markdown, with layout regions typed
  (header, title, text, table, image, image_caption) and figures extracted as crops.
  No page ever leaves the machine.
- **PDFs, images, or a directory of scans.** Multi-page runs share one 32k context, so a
  table split across a page break resolves in a single pass.
- **A Claude Science skill** (`skill/`) exposing `ocr_pdf`, `ocr_image`, `ocr_summarize`,
  `ocr_status`, `ocr_pages` in the agent's Python kernel.

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon (M1 or later), 16 GB unified memory recommended |
| OS | macOS 14+ (MPS requirement) |
| Python | 3.10–3.13 (3.13 is what this was built and tested against) |
| Disk | ~8 GB (6.7 GB weights + ~1 GB venv) |

No CUDA, no Docker, no `sudo`. The installer touches only its own directory.

## Install

```bash
git clone https://github.com/markwgreenlee/unlimited-ocr-macos.git ~/Unlimited-OCR
cd ~/Unlimited-OCR
./install.sh
```

Cloning to `~/Unlimited-OCR` means the skill's default path just works; anywhere else is
fine if you set `UNLIMITED_OCR_ROOT`. The installer checks the platform, builds `.venv`,
installs the pinned dependency set, downloads the weights, runs the Metal correctness probe,
and finishes with an end-to-end transcription so you know it works before you trust it.

The weight download is the slow part (6.7 GB, ~17 minutes on a 60 Mbit line).

## Usage

```bash
./run-ocr.sh page.png                      # single image, tiled "gundam" mode
./run-ocr.sh scan.pdf --out out/scan        # PDF — every page in one pass
./run-ocr.sh ~/scans/                       # a directory of page images
./run-ocr.sh page.png --mode base           # one 1024 px view, faster on simple pages
./run-ocr.sh figure.png --prompt "<image>Parse the figure."
./run-ocr.sh page.png --device cpu          # force CPU
./run-ocr.sh --help
```

Each run writes into the output directory:

| file | contents |
|---|---|
| `result.md` | the parsed document as Markdown |
| `result_with_boxes.jpg` | input image with detected layout regions drawn |
| `images/*.jpg` | figure and table crops referenced from `result.md` |
| `pages/page_XXXX.png` | rasterised pages (PDF input only) |

**Modes.** `gundam` (default for single images) uses a 1024 px global view plus 640 px
tiles — best for dense pages and large scans. `base` uses one 1024 px view and is faster on
clean, simple pages. Multi-page input always uses the 1024 px view, since crop mode isn't
supported there.

**Prompts.** The model follows a handful of trained instructions and the leading `<image>`
token is required: `<image>document parsing.`, `<image>Multi page parsing.`,
`<image>Free OCR.`, `<image>Parse the figure.`

## Use as a Claude Science skill

`skill/` contains `SKILL.md` and a `kernel.py` sidecar. Three steps, and the first is the
one people miss:

**1. Grant Claude Science access to this directory.** Its kernels are sandboxed and cannot
reach arbitrary paths in your home directory. The skill launches `./.venv/bin/python` as a
subprocess and reads `./model`, so without a grant the helpers fail at the first call. Ask
the agent for it — *"request access to ~/Unlimited-OCR"* — and approve the read-write
prompt, or add the folder in the host-access settings.

**2. Create the skill.** Ask the agent:

> Read `~/Unlimited-OCR/skill/SKILL.md` and `~/Unlimited-OCR/skill/kernel.py` and publish
> them as a skill named `unlimited-ocr`.

It writes both files with `host.skills.edit` and publishes with `host.skills.publish`. You
can also paste the two files' contents into the conversation instead, if you'd rather not
grant access before the skill exists — but step 1 is still required for the skill to *run*.
If your organisation has custom skills turned off, publishing is refused; in that case the
agent can still `exec(open(".../skill/kernel.py").read())` to get the helpers for one
session, or just drive `run-ocr.sh` through the shell.

**3. Tell it where the install lives — unless you cloned to the default.** The sidecar
resolves the root as: an explicit `root=` argument, then `$UNLIMITED_OCR_ROOT`, then
`~/Unlimited-OCR`. Note that an agent kernel does **not** inherit your login shell's
environment (it gets a minimal one), so `export UNLIMITED_OCR_ROOT=...` in your terminal
does not reach it. If you cloned somewhere other than `~/Unlimited-OCR`, either pass
`root="/path/to/install"` to the helpers or have the agent set
`os.environ["UNLIMITED_OCR_ROOT"]` once per session.

Verify with `ocr_status()` — it should report `ok: True` and `device: "mps"`. After that,
"OCR this scan and summarize it" works in one step, and these land in the Python kernel:

| function | purpose |
|---|---|
| `ocr_status()` | verify the install, report `mps` or `cpu` |
| `ocr_pdf(pdf, pages_per_chunk=12, dpi=300)` | rasterise, chunk, OCR, stitch → Markdown |
| `ocr_image(img, mode="gundam")` | one image → Markdown + layout overlay |
| `ocr_summarize(markdown, focus=…)` | parallel per-section notes, then one synthesis |
| `ocr_pages(pdf, …)` | rasterise only |

`ocr_pdf` returns per-chunk character counts, timings and exit codes, and an
`empty_chunks` list — which matters, because an empty result is this model's failure mode
(see below) rather than an exception.

## How the port works

The model's remote code (`trust_remote_code`) is CUDA-specific in three ways. Rather than
patch the downloaded files — which `transformers` re-fetches into its own module cache, and
which a weight re-download would clobber — `ocr.py` installs a runtime shim:

1. **`.cuda()` on every input tensor** → redirected to MPS (or CPU) by rebinding
   `Tensor.cuda` / `Module.cuda`.
2. **`torch.autocast("cuda", dtype=bfloat16)`** → neutralised. Weights and image tensors are
   already a single dtype, so autocast is a no-op for inference.
3. **Hard-coded `.to(torch.bfloat16)` on image tensors** (4 call sites) → redirected to the
   active dtype, so `--dtype fp16` doesn't die with *"Input type (c10::BFloat16) and bias
   type (c10::Half) should be the same"*.

Two further requirements worth knowing if you fork this:

- The model **must** be loaded with `attn_implementation="eager"`. Its `ATTENTION_CLASSES`
  table has only eager and flash-attention entries — no `sdpa` — so the default raises
  `KeyError: 'mha_sdpa'`.
- `max_length` counts *prompt* tokens. A single 1024 px view is already 277 input tokens, so
  a small cap raises a `ValueError` before inference starts.

## The MPS bug

On torch 2.10 / macOS 27, `Tensor.masked_scatter_` with a **broadcast mask** silently writes
nothing on MPS. The model uses exactly that pattern to insert vision features into the token
embeddings:

```python
inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), image_features)
```

A `(seq, 1)` mask against a `(seq, hidden)` block. On Metal the write evaporates, the decoder
receives placeholder embeddings where the page should be, and it emits end-of-sequence
immediately. **The run reports success, writes `result_with_boxes.jpg`, and leaves
`result.md` empty** — no error, no warning, and no NaNs anywhere, which is what makes it
hard to diagnose.

`ocr.py` routes that single write through CPU (`masked_scatter_via_cpu`). It's one small
tensor per image, so the cost is negligible.

`probe-mps-ops.py` compares 12 primitives CPU-vs-MPS and is how you check the state of this:

```bash
./.venv/bin/python probe-mps-ops.py            # the real check
./.venv/bin/python probe-mps-ops.py --device cpu   # self-test; every row must read ok
```

Expected output flags 4 disagreements, and that is **correct**:

- `masked_scatter_ (contiguous)` and `(via view)` — the unpatched bug, kept as a regression
  detector. **When these turn green, the workaround can be deleted.**
- `lm_head matmul` (~1.4% relative) and `topk(6) expert routing` (~1% of near-tied random
  values) — bf16 accumulating in a different order on Metal.

The rows that must read `ok` are `masked_scatter_ (patched)` and
`masked_scatter_ (shim installed)`. Those bf16 precision differences provably don't change
the output: MPS and CPU produce **byte-identical** transcriptions of the same page (verified
by md5).

`diagnose-mps.py` is the heavier tool — it hooks every submodule to find the first whose
output goes non-finite, per device and dtype:

```bash
./.venv/bin/python diagnose-mps.py --configs mps:bf16,mps:fp16,cpu:bf16
```

## Verified results

MacBook Air, Apple M5 (10 CPU / 8 GPU cores), 16 GB, macOS 27, torch 2.10.0:

| task | device | time | output |
|---|---|---|---|
| paper page, gundam mode | MPS | 49 s | 1976 chars |
| same page, gundam mode | CPU | 131 s | 1976 chars, byte-identical |
| 1000×320 text image, base mode | CPU | 58 s | exact transcription |
| 2-page PDF @150 dpi, multi-page | CPU | 181 s | both pages |
| model load | MPS | 10 s | |

Budget roughly 20–60 s per page on MPS depending on density.

## Troubleshooting

**`result.md` is empty.** The Metal bug above. Run `probe-mps-ops.py` and confirm
`masked_scatter_ (shim installed)` reads `ok`; check the same page with `--device cpu`.

**Three warnings on every run** — `position_ids ... newly initialized`, *"You should
probably TRAIN this model"*, `attention mask ... not set` / `Setting pad_token_id to
eos_token_id:1`. All benign, all appear on correct runs. `position_ids` is a fixed index
buffer that newer `transformers` rebuilds rather than a learned weight; the attention mask
only matters for padded batches, and single-sequence inference implies a correct all-ones
mask.

**Out of memory / heavy swapping.** bf16 weights are 6.7 GB of unified memory. Don't use
`--dtype fp32` (13.4 GB). Close other large applications, or use `--device cpu`.

**Long PDFs.** One invocation shares a 32768-token context, so ~10–20 pages is the ceiling.
The skill's `ocr_pdf` chunks automatically; from the CLI, split the PDF yourself.

## Files

| path | what it is |
|---|---|
| `install.sh` | installer: platform checks, venv, weights, verification |
| `ocr.py` | CLI + the CUDA→MPS shim and the `masked_scatter_` workaround |
| `run-ocr.sh` | launcher (activates `.venv`, sets `PYTORCH_ENABLE_MPS_FALLBACK=1`) |
| `probe-mps-ops.py` | CPU-vs-MPS comparison of 12 primitives; self-tests with `--device cpu` |
| `diagnose-mps.py` | per-module non-finite hunt across device/dtype configurations |
| `skill/SKILL.md`, `skill/kernel.py` | the Claude Science skill |
| `samples/make_sample_page.py` | generates a synthetic test page |

## Licence and attribution

This port is MIT licensed (see `LICENSE`).

The model and upstream code are MIT licensed by Baidu Inc. **Weights are not redistributed
here** — `install.sh` downloads them from
[huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR). If you use
the model in published work, cite the upstream paper:

> Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing. Baidu Inc.,
> [arXiv:2606.23050](https://arxiv.org/abs/2606.23050)

Unlimited-OCR builds on [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR).
