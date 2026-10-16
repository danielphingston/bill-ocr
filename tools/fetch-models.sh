#!/usr/bin/env bash
# Download the ONNX export for the Node tests and the optional 8-bit build. The web page doesn't need this.
set -euo pipefail
dir="$(dirname "$0")/../models/falcon-ocr"
mkdir -p "$dir"
for f in tokenizer.json config.json falcon_ocr_kv_token.onnx; do
  [ -s "$dir/$f" ] || curl -fL --progress-bar -o "$dir/$f" "https://huggingface.co/ningpp/Falcon-OCR-ONNX/resolve/46f25593bc93c19003fbc6b9c968c137476be4c5/$f"
done
# 8-bit build (tools/quantize.py), hosted in a separate repo. Non-fatal if not uploaded yet.
[ -s "$dir/falcon_ocr_kv_token_q8.onnx" ] || curl -fL --progress-bar -o "$dir/falcon_ocr_kv_token_q8.onnx" \
  "https://huggingface.co/<you>/Falcon-OCR-ONNX-Q8/resolve/main/falcon_ocr_kv_token_q8.onnx" \
  || echo "skipping falcon_ocr_kv_token_q8.onnx (repo not uploaded yet?)"
echo "models in $dir"
