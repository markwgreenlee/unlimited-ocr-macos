import glob
import json
import os
import subprocess
import time

OCR_ROOT_DEFAULT = "~/Unlimited-OCR"
OCR_ROOT_ENV = "UNLIMITED_OCR_ROOT"
CHUNK_PAGES = 12
MAP_CHARS = 12000


def ocr_install_root(root=None):
    """Resolve the install directory: explicit arg, then $UNLIMITED_OCR_ROOT, then the default."""
    if root:
        return os.path.expanduser(root)
    return os.path.expanduser(os.environ.get(OCR_ROOT_ENV, OCR_ROOT_DEFAULT))


def ocr_status(root=None):
    """Report on the local Unlimited-OCR install: paths, weights, active device."""
    root = ocr_install_root(root)
    py = os.path.join(root, ".venv", "bin", "python")
    weights = os.path.join(root, "model", "model-00001-of-000001.safetensors")
    out = {"root": root, "python": py,
           "python_ok": os.path.exists(py),
           "weights_gb": round(os.path.getsize(weights) / 1e9, 2) if os.path.exists(weights) else 0.0,
           "driver_ok": os.path.exists(os.path.join(root, "ocr.py"))}
    if out["python_ok"]:
        probe = "import torch;print(torch.__version__, torch.backends.mps.is_available())"
        r = subprocess.run([py, "-c", probe], capture_output=True, text=True)
        parts = r.stdout.split()
        out["torch"] = parts[0] if parts else r.stderr.strip()[:200]
        out["mps"] = len(parts) > 1 and parts[1] == "True"
        out["device"] = "mps" if out.get("mps") else "cpu"
    out["ok"] = out["python_ok"] and out["driver_ok"] and out["weights_gb"] > 6
    return out


def ocr_pages(pdf_path, out_dir=None, dpi=300, first_page=None, last_page=None, root=None):
    """Rasterise a PDF to page PNGs with the install's PyMuPDF. Returns page paths."""
    root = ocr_install_root(root)
    if out_dir is None:
        out_dir = os.path.join("ocr_out", os.path.splitext(os.path.basename(pdf_path))[0])
    pages_dir = os.path.join(out_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    script = os.path.join(out_dir, "_rasterise.py")
    with open(script, "w") as fh:
        fh.write("import sys, fitz\n"
                 "src, dst, dpi, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])\n"
                 "doc = fitz.open(src); mat = fitz.Matrix(dpi/72, dpi/72)\n"
                 "hi = doc.page_count if hi < 0 else min(hi, doc.page_count)\n"
                 "for i in range(lo-1, hi):\n"
                 "    doc[i].get_pixmap(matrix=mat).save('%s/page_%04d.png' % (dst, i+1))\n"
                 "print(hi-lo+1)\n")
    r = subprocess.run([os.path.join(root, ".venv", "bin", "python"), script, pdf_path, pages_dir,
                        str(dpi), str(first_page or 1), str(last_page if last_page else -1)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("rasterise failed: " + r.stderr[-400:])
    return sorted(glob.glob(os.path.join(pages_dir, "page_*.png")))


def ocr_pdf(pdf_path, out_dir=None, pages_per_chunk=None, dpi=300, device="auto",
            first_page=None, last_page=None, prompt=None, root=None, verbose=True):
    """OCR a PDF in page chunks and return the combined Markdown.

    Every page in a chunk shares one 32k context, which is what lets a table
    spanning a page break resolve; chunking keeps long documents inside that
    limit. Expect roughly 20-60 s per page, so run this in a background cell.
    """
    root = ocr_install_root(root)
    if pages_per_chunk is None:
        pages_per_chunk = CHUNK_PAGES
    if out_dir is None:
        out_dir = os.path.join("ocr_out", os.path.splitext(os.path.basename(pdf_path))[0])
    started = time.time()
    pages = ocr_pages(pdf_path, out_dir, dpi, first_page, last_page, root)
    chunks = [pages[i:i + pages_per_chunk] for i in range(0, len(pages), pages_per_chunk)]
    py = os.path.join(root, ".venv", "bin", "python")
    sections, records = [], []
    for k, chunk in enumerate(chunks, 1):
        cdir = os.path.join(out_dir, "chunk_%02d" % k)
        idir = os.path.join(cdir, "input")
        os.makedirs(idir, exist_ok=True)
        for src in chunk:
            dst = os.path.join(idir, os.path.basename(src))
            if not os.path.exists(dst):
                os.link(src, dst)
        # absolute paths: the driver runs with cwd=root, not the caller's cwd
        cmd = [py, os.path.join(root, "ocr.py"), os.path.abspath(idir),
               "--out", os.path.abspath(cdir), "--device", device]
        if prompt:
            cmd += ["--prompt", prompt]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
        md_path = os.path.join(cdir, "result.md")
        md = open(md_path, encoding="utf-8").read() if os.path.exists(md_path) else ""
        lo = os.path.basename(chunk[0]).split("_")[1].split(".")[0]
        hi = os.path.basename(chunk[-1]).split("_")[1].split(".")[0]
        records.append({"chunk": k, "pages": "%s-%s" % (lo, hi), "chars": len(md),
                        "seconds": round(time.time() - t0, 1),
                        "returncode": r.returncode, "stderr_tail": r.stderr[-300:] if r.returncode else ""})
        sections.append("<!-- pages %s-%s -->\n\n%s" % (lo, hi, md.strip()))
        if verbose:
            print("chunk %d/%d pages %s-%s: %d chars in %.0fs"
                  % (k, len(chunks), lo, hi, len(md), records[-1]["seconds"]))
    document = "\n\n".join(sections).strip() + "\n"
    doc_path = os.path.join(out_dir, "document.md")
    with open(doc_path, "w", encoding="utf-8") as fh:
        fh.write(document)
    result = {"markdown": document, "document_md": doc_path, "out_dir": out_dir,
              "page_count": len(pages), "chunks": records, "empty_chunks": [c["chunk"] for c in records if c["chars"] == 0],
              "seconds": round(time.time() - started, 1)}
    with open(os.path.join(out_dir, "ocr_run.json"), "w") as fh:
        json.dump({k: v for k, v in result.items() if k != "markdown"}, fh, indent=2)
    return result


def ocr_image(image_path, out_dir=None, mode="gundam", device="auto", prompt=None, root=None):
    """OCR a single image (a scan, screenshot, figure). Returns the Markdown."""
    root = ocr_install_root(root)
    if out_dir is None:
        out_dir = os.path.join("ocr_out", os.path.splitext(os.path.basename(image_path))[0])
    cmd = [os.path.join(root, ".venv", "bin", "python"), os.path.join(root, "ocr.py"),
           os.path.abspath(image_path), "--out", os.path.abspath(out_dir),
           "--mode", mode, "--device", device]
    if prompt:
        cmd += ["--prompt", prompt]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    md_path = os.path.join(out_dir, "result.md")
    if not os.path.exists(md_path):
        raise RuntimeError("no result.md; stderr tail: " + r.stderr[-400:])
    md = open(md_path, encoding="utf-8").read()
    return {"markdown": md, "result_md": md_path, "out_dir": out_dir,
            "boxes_jpg": os.path.join(out_dir, "result_with_boxes.jpg")}


def ocr_summarize(markdown, focus=None, map_chars=None, model=None):
    """Map-reduce summary of OCR'd Markdown: per-section notes, then one synthesis.

    Short documents need no map step — read the Markdown directly instead.
    """
    if map_chars is None:
        map_chars = MAP_CHARS
    if model is None:
        model = host.reasoning_model()
    aim = focus or "the document's purpose, methods, key quantitative results, and conclusions"
    blocks, buf = [], ""
    for para in markdown.split("\n\n"):
        if len(buf) + len(para) > map_chars and buf:
            blocks.append(buf)
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        blocks.append(buf)
    if len(blocks) == 1:
        notes = [blocks[0]]
    else:
        reqs = [{"prompt": "Extract the factual content of this section of an OCR'd document. "
                           "Keep every number, unit, name and citation verbatim; note any text that "
                           "looks garbled by OCR. No preamble.\n\nSECTION %d/%d:\n%s"
                           % (i + 1, len(blocks), b)} for i, b in enumerate(blocks)]
        got = host.llm(reqs, max_concurrency=6)
        notes = [g.get("text", "") if isinstance(g, dict) else "" for g in got]
    synth = host.llm({"prompt": "You are summarising an OCR'd document from these ordered section "
                               "notes. Focus on %s. Write: a 2-3 sentence abstract, then 'Key points' "
                               "as bullets with concrete numbers, then 'Caveats' noting anything the "
                               "OCR may have garbled or that was unreadable. Do not invent content.\n\n%s"
                               % (aim, "\n\n---\n\n".join(notes)), "model": model})
    return {"summary": synth.get("text", ""), "section_notes": notes,
            "sections": len(blocks), "chars_in": len(markdown)}
