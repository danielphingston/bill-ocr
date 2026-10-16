# Falcon OCR bill reader (in-browser)

Upload bills, receipts or invoices and get structured JSON back. Falcon-OCR runs in the browser
with ONNX Runtime Web on WebGPU. No backend: images never leave the device.

> **Experimental proof of concept:** This project is a prototype, and results are not reliable
> across many real-world documents. OCR can omit or misread content, and extracted fields or line
> items may be incomplete or incorrect. Review the raw OCR and verify all results manually.

## Run it

```bash
npm run serve        # any static file server works; this one is http-server on :8080
# open http://localhost:8080
```

You still need a static host because browsers block module workers and `fetch` on `file://`.
Any static host works (GitHub Pages, Cloudflare Pages, S3, nginx). Nothing runs server-side.

1. Click **Load model**. It downloads the weights from Hugging Face once and then caches them
   in the browser (Cache Storage), so later visits work offline.
2. Drop, paste or pick images. They go through the model one at a time, and the text streams in as it is generated.
3. Each result has tabs: **Fields**, **Line items**, **Raw OCR** and **JSON**. Use **Export all** to download JSON or a CSV summary.

### Browser requirements

- Needs WebGPU: Chrome or Edge 113+ on desktop, or recent Safari/Firefox with WebGPU on.
  Without it the page falls back to WASM on the CPU, which is very slow.
- **Laptops with two GPUs (Windows):** Chrome and Edge give WebGPU the integrated GPU and ignore the page's
  `high-performance` request ([crbug 369219127](https://issues.chromium.org/issues/369219127)). The badge at the top shows the GPU in use.
  To switch, enable `chrome://flags/#force-high-performance-gpu` (or `edge://flags/...`), or set the browser
  to *High performance* in Windows Settings → System → Display → Graphics. Then fully restart the browser.
- GPU memory: prefill uses dense attention, so it needs about `16 × S² × 4` bytes per attention
  tensor (S = image tokens). Balanced resolution (≤ ~1,960 tokens) needs ~250 MB per tensor.
  High (≤ ~3,920 tokens) needs ~1 GB per tensor, which only works on GPUs with plenty of memory.

## Where the model comes from

Nothing is stored in the repo. When you click **Load model**, the page downloads the weights straight from
Hugging Face (the CDN allows cross-origin requests) and keeps them in the browser's Cache Storage.

- TII's official [tiiuae/Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) only ships PyTorch weights, which a browser can't run.
- The page uses the community ONNX export [ningpp/Falcon-OCR-ONNX](https://huggingface.co/ningpp/Falcon-OCR-ONNX)
  (`falcon_ocr_kv_token.onnx`, fp32, 1.08 GB) and its `tokenizer.json`. That export was checked against
  the PyTorch model (Top-5 logits match) and matched it on our test images.

| Variant | Source | Size | Notes |
|---|---|---|---|
| 8-bit (default when available) | [acid1burn/falcon_ocr_kv_token_q8.onnx](https://huggingface.co/acid1burn/falcon_ocr_kv_token_q8.onnx) | 450 MB | INT8 weight-only ONNX. Falls back to fp32 if the Q8 file is unavailable. |
| fp32 | Hugging Face | 1.08 GB | Fallback model; downloads once per browser. |

A 4-bit build (340 MB) is possible but not offered: it looped on Chinese text and dropped a quantity on the receipt.

To build the 8-bit file:

```bash
tools/fetch-models.sh                       # downloads the fp32 export into models/ (gitignored)
pip install onnx==1.20.1 onnxruntime onnx_ir
python3 -I tools/quantize.py models/falcon-ocr/falcon_ocr_kv_token.onnx models/falcon-ocr/falcon_ocr_kv_token_q8.onnx 8 32
```

Embeddings stay fp32 because an fp16 Gather needs the WebGPU `shader-f16` feature, which not every GPU has.
`--emb-fp16` saves 100 MB if you only target GPUs that support it. The page can also load any `.onnx` file picked from disk.

## Layout

| Path | What |
|---|---|
| `index.html` | UI |
| `src/app.js` | Queue, rendering, exports |
| `src/worker.js` | Web Worker that owns the ORT session, downloads and caches the model, decodes images |
| `src/falcon.js` | Port of Falcon-OCR preprocessing (Pillow-exact bicubic resize, patching, 2D positions, hybrid attention mask), the KV-cache greedy decode loop, and the byte-level BPE detokenizer |
| `src/extract.js` | Rule-based field extraction: vendor, number, date, currency, line items, totals, payment, and sanity checks. Handles plain text, Markdown and HTML tables. |
| `test/run-node.mjs` | Same pipeline on onnxruntime-node (CPU) |
| `test/extract-test.mjs` | Extractor assertions |
| `test/browser-e2e.mjs` | Drives the page in headless Chromium (WebGPU via SwiftShader) |

## Test

```bash
npm install
tools/fetch-models.sh                   # Node tests need the weights locally
node test/extract-test.mjs
node test/run-node.mjs test/images/bill.png plain balanced          # fp32
MODEL=models/falcon-ocr/falcon_ocr_kv_token_q8.onnx node test/run-node.mjs test/images/bill.png
npm run serve & node test/browser-e2e.mjs test/images/bill.png fp32 400
```

## Known limits

- **Prototype reliability:** This is a proof of concept, not a production-ready document-processing system. Performance and accuracy vary by document; many layouts may not be read or structured correctly. Small text, dense tables, unusual formatting, and low-quality images are common failure cases.
- **Resolution matters.** At 768px the synthetic receipt came back as hallucinated JSON
  (CORD-style `menu/nm/cnt` keys). At native resolution it was read word for word. The presets therefore
  stay at or near the reference settings (1536px / ~1M px). If the model does return JSON, it is
  parsed into `model_json`.
- **Field extraction is heuristic.** It uses regexes and table parsing, not a language model, so
  fields and line items may be missing or misclassified. Check all values, especially totals, before use.
  The `checks` block flags some subtotals and line items that don't add up, but cannot catch every error.
  The next step up would be a small in-browser LLM (e.g. Qwen3-0.6B via transformers.js) that maps the OCR text to a JSON schema. That adds ~500 MB to the download.
- The ONNX graph stores the KV cache after head expansion (16 heads), about 180 KB per token. Each decode step
  concatenates the cache, so decode gets slower as outputs get longer.
