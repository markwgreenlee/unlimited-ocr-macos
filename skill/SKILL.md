---
name: unlimited-ocr
description: "Read scanned or image-based PDFs and document images into Markdown with the locally installed Unlimited-OCR vision-language model (~/Unlimited-OCR), then summarize them. Use this skill whenever the user wants text, tables, figures or a summary out of a PDF, scan, screenshot or photographed page — including phrases like OCR, read this paper, extract the text, digitise, parse this document, what does this scan say, or summarize this PDF — and especially when a PDF has no extractable text layer, when pdfplumber/PyMuPDF text extraction returns empty or garbled output, or when layout structure (tables, figure captions, reading order) matters. Prefer this over plain text extraction for scans, photographs of pages, and any PDF where the text layer is missing or unreliable."
---

# Unlimited-OCR (local, Apple Silicon)

Baidu's Unlimited-OCR is installed at `~/Unlimited-OCR` — a DeepSeek-OCR-derived
vision-language model (6.7 GB, bf16) that transcribes a page image into Markdown and labels
layout regions (header, title, text, table, image, image_caption). It runs on the Mac's GPU
via MPS. It reads pixels, so it works on scans and photographs where there is no text layer
to extract.

**It is a model, not a text extractor.** When a PDF already has a good text layer, plain
extraction (PyMuPDF `page.get_text()`) is instant and exact — use that instead. Reach for
this skill when extraction comes back empty, garbled, or loses the table and reading-order
structure you need.

## Helpers

Loading this skill defines these in your Python kernel:

| function | purpose |
|---|---|
| `ocr_status()` | verify the install and report the active device (`mps` or `cpu`) |
| `ocr_pdf(pdf, out_dir=…, pages_per_chunk=12, dpi=300, device="auto")` | OCR a PDF in page chunks → combined Markdown |
| `ocr_image(img, mode="gundam")` | OCR one image → Markdown + layout-box overlay |
| `ocr_summarize(markdown, focus=…)` | map-reduce summary of long OCR output |
| `ocr_pages(pdf, …)` | rasterise pages only, without OCR |

All take `root=`; otherwise they resolve `$UNLIMITED_OCR_ROOT`, defaulting to `~/Unlimited-OCR`.

## Workflow

1. **Check the text layer first.** `fitz.open(pdf)[0].get_text()` in a `python` cell. If it
   returns clean text, say so and extract normally — running a 6.7 GB model on a page that
   already has its text is a waste of the user's time. If it's empty or garbled, continue.
2. **Confirm the install** with `ocr_status()`. `ok: True` with `device: "mps"` is the
   healthy state. `device: "cpu"` means Metal isn't reachable and the run will be ~2.7×
   slower — still correct, worth telling the user before a long document.
3. **Run the OCR.** `res = ocr_pdf("scan.pdf", out_dir="ocr_out/scan")`. Budget roughly
   **20–60 s per page**, so anything past a couple of pages belongs in a `background: true`
   cell — keep working and pick up the result when it lands. `res["markdown"]` is the text,
   `res["document_md"]` the path, `res["chunks"]` the per-chunk timing and character counts.
4. **Check for empty chunks.** `res["empty_chunks"]` lists chunks that produced nothing.
   That is the model's failure mode — it returns no tokens rather than erroring. One empty
   chunk among many usually means a blank or near-blank page; if *all* chunks are empty,
   something is wrong with the install rather than the document (see Troubleshooting).
5. **Save the Markdown as an artifact** with `save_artifacts` before summarizing — the
   transcription is the durable deliverable and the user may want to check the summary
   against it.
6. **Summarize.** For a few pages, just read `res["markdown"]` and write the summary
   yourself — you have the whole document in context and will do better than a map-reduce.
   For long documents call `ocr_summarize(res["markdown"], focus="…")`, which takes notes
   per section in parallel and then synthesises; pass `focus=` to steer it (e.g.
   `focus="the statistical methods and effect sizes"`).
7. **Quote OCR'd numbers carefully.** Everything came out of a model, so a digit can be
   wrong in a way that is invisible in fluent prose. When a specific figure matters —
   a dose, an N, a p-value, a date — say it came from OCR, and for anything critical offer
   to render that page so the user can check it against the original.

## Page chunking

Every page in a chunk shares one 32768-token context. That is a feature: a table split
across a page break resolves correctly because the model sees both halves at once. It also
sets the ceiling — roughly 10–20 pages per chunk, hence the default of 12. Lower
`pages_per_chunk` for dense pages (small print, big tables); chunk boundaries are the only
place cross-page structure can be lost, so try to put them where a section ends.

`dpi=300` is the default and suits most scans. Drop to 150 for speed on clean, large-print
documents; raise it for small print or footnotes.

## Prompts and modes

The model follows a handful of trained instructions, and the leading `<image>` token is
required. Pass with `prompt=`:

- `"<image>document parsing."` — default for single images; full layout + Markdown
- `"<image>Multi page parsing."` — default for multi-page runs
- `"<image>Free OCR."` — plain text, no layout structure
- `"<image>Parse the figure."` — for charts and diagrams

`ocr_image` takes `mode="gundam"` (1024 px view plus 640 px tiles — best for dense pages)
or `mode="base"` (one 1024 px view — faster on simple pages). It also writes
`result_with_boxes.jpg`, the input with detected regions drawn; embed it when the user asks
what the model saw or doubts a transcription.

## Troubleshooting

**Every chunk empty.** The install's `ocr.py` carries a workaround for an MPS bug where
`masked_scatter_` with a broadcast mask silently writes nothing, which starves the decoder
of the image and makes it emit end-of-sequence immediately — the run "succeeds" with zero
output. Verify with `~/Unlimited-OCR/.venv/bin/python probe-mps-ops.py`: the
`masked_scatter_ (shim installed)` row must read `ok`. Confirm the same page works with
`device="cpu"`; if CPU works and MPS doesn't, the workaround has regressed.

**Warnings on every run** — `position_ids ... newly initialized`, `attention mask ... not
set`, `Setting pad_token_id to eos_token_id`. All three are benign and appear on correct
runs too. Don't report them as errors.

**Long document, slow.** Keep it in a background cell and report progress per chunk; the
helper prints a line as each chunk finishes. Splitting into several `ocr_pdf` calls over
page ranges (`first_page`/`last_page`) lets you show the user the first pages early.

`~/Unlimited-OCR/README-macos.md` documents the install itself: layout, pinned
versions, the MPS issue, and the CPU-vs-MPS verification.
