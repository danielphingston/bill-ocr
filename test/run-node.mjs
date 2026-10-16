// End-to-end check of src/falcon.js with onnxruntime-node (CPU).
// Usage: node test/run-node.mjs [image.png] [category] [preset] [--dump out.raw]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ort from 'onnxruntime-node';
import { PNG } from 'pngjs';
import { preprocess, generate, Detokenizer, cleanOutput, rgbaToRgb } from '../src/falcon.js';
import { extractBill } from '../src/extract.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const dumpAt = args.indexOf('--dump');
const dump = dumpAt >= 0 ? args.splice(dumpAt, 2)[1] : null;
const [imgPath = path.join(root, 'test/images/bill.png'), category = 'plain', preset = 'balanced'] = args;
const modelPath = process.env.MODEL || path.join(root, 'models/falcon-ocr/falcon_ocr_kv_token.onnx');

const png = PNG.sync.read(fs.readFileSync(imgPath));
const rgb = rgbaToRgb(png.data, png.width, png.height);
const pre = preprocess(rgb, png.width, png.height, { category, preset });
console.log(`image ${png.width}x${png.height} -> ${pre.width}x${pre.height}, grid ${pre.rows}x${pre.cols}, prompt ${pre.S} tokens`);
if (dump) {
  fs.writeFileSync(dump, pre.resized);
  console.log(`dumped resized RGB to ${dump}`);
  if (process.env.DUMP_ONLY) process.exit(0);
}

const detok = new Detokenizer(JSON.parse(fs.readFileSync(path.join(path.dirname(modelPath), 'tokenizer.json'), 'utf8')));
const t0 = performance.now();
const session = await ort.InferenceSession.create(modelPath, { graphOptimizationLevel: 'all' });
console.log(`session ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const res = await generate(ort, session, pre, {
  maxNewTokens: Number(process.env.MAX_NEW || 1024),
  onPrefill: (ms) => console.log(`prefill ${(ms / 1000).toFixed(2)}s`),
  onToken: (ids) => { if (ids.length % 25 === 0) process.stdout.write(`.${ids.length}`); },
});
const text = cleanOutput(detok.decode(res.ids));
console.log(`\ndecode ${res.ids.length} tokens in ${(res.decodeMs / 1000).toFixed(2)}s (${(res.ids.length / (res.decodeMs / 1000)).toFixed(1)} tok/s)`);
console.log('<<<OCR\n' + text + '\nOCR>>>');
console.log(JSON.stringify(extractBill(text), null, 2));
