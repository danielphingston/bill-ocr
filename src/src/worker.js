// Inference worker: owns the ONNX session so the page never freezes.
// Messages in:  {type:'load', model, tokenizer, device}  {type:'ocr', id, blob, category, preset, maxNewTokens}  {type:'abort'}
// Messages out: {type:'progress'|'ready'|'status'|'token'|'done'|'error', ...}
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';
import { preprocess, generate, Detokenizer, cleanOutput, rgbaToRgb } from './falcon.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

const CACHE = 'falcon-ocr-models-v2';
let session = null;
let detok = null;
let abort = false;

const post = (msg) => self.postMessage(msg);

// Fetch with download progress; cache in Cache Storage so reloads are offline.
// `source` may be a Blob (file picked from disk), a URL, or a list of URLs to try in order.
async function fetchCached(source, label) {
  if (source instanceof Blob) return new Uint8Array(await source.arrayBuffer());
  if (Array.isArray(source)) {
    let err;
    for (const s of source) {
      try { return await fetchCached(s, label); } catch (e) { err = e; }
    }
    throw err;
  }
  const url = new URL(source, self.location.href).href;
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* storage blocked */ }
  const hit = cache && (await cache.match(url));
  if (hit) {
    post({ type: 'progress', label, loaded: 1, total: 1, cached: true });
    return new Uint8Array(await hit.arrayBuffer());
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - last > 150) { post({ type: 'progress', label, loaded, total }); last = now; }
  }
  post({ type: 'progress', label, loaded, total: total || loaded });
  const blob = new Blob(chunks);
  if (cache) {
    try { await cache.put(url, new Response(blob, { headers: { 'content-type': 'application/octet-stream' } })); }
    catch (e) { post({ type: 'status', text: `Not cached (${e.name}); it will download again next time.` }); }
  }
  return new Uint8Array(await blob.arrayBuffer());
}

async function load({ model, tokenizer, device }) {
  if (session) { await session.release(); session = null; }
  const tokBytes = await fetchCached(tokenizer, 'tokenizer');
  detok = new Detokenizer(JSON.parse(new TextDecoder().decode(tokBytes)));
  const modelBytes = await fetchCached(model, 'model');
  post({ type: 'status', text: `Compiling model on ${device.toUpperCase()}…` });
  const t0 = performance.now();
  const opts = { executionProviders: [device], graphOptimizationLevel: 'all' };
  let adapterInfo = null;
  if (device === 'webgpu') {
    const gpu = await highPerformanceDevice();
    adapterInfo = gpu.info;
    opts.executionProviders = [{ name: 'webgpu', device: gpu.device }];
    // Keep the KV cache on the GPU between decode steps.
    opts.preferredOutputLocation = { present_key_values: 'gpu-buffer', next_token: 'cpu' };
  }
  session = await ort.InferenceSession.create(modelBytes, opts);
  post({ type: 'ready', ms: performance.now() - t0, device, adapter: adapterInfo });
}

// ORT's native WebGPU EP picks its own adapter and ignores env.webgpu.powerPreference,
// so request the discrete GPU ourselves and hand ORT the device. Chrome on Windows
// ignores powerPreference too (crbug.com/369219127); there only a browser flag helps.
let gpuDevice = null;
async function highPerformanceDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter available');
  const requiredLimits = {};
  for (const k in adapter.limits) if (typeof adapter.limits[k] === 'number') requiredLimits[k] = adapter.limits[k];
  gpuDevice?.destroy();
  gpuDevice = await adapter.requestDevice({ requiredLimits, requiredFeatures: [...adapter.features] });
  gpuDevice.lost.then((e) => { if (e.reason !== 'destroyed') post({ type: 'error', text: `GPU device lost: ${e.message || e.reason}` }); });
  const i = adapter.info || {};
  return { device: gpuDevice, info: { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description } };
}

async function decodeImage(blob) {
  const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const { data } = ctx.getImageData(0, 0, bmp.width, bmp.height);
  bmp.close();
  return { rgb: rgbaToRgb(data, canvas.width, canvas.height), width: canvas.width, height: canvas.height };
}

async function ocr({ id, blob, category, preset, maxNewTokens }) {
  if (!session) throw new Error('Model not loaded');
  abort = false;
  const img = await decodeImage(blob);
  const pre = preprocess(img.rgb, img.width, img.height, { category, preset });
  post({ type: 'status', id, text: `Reading ${pre.width}×${pre.height} (${pre.S} tokens)…`, prompt: pre.S, size: [pre.width, pre.height] });
  let lastSent = 0;
  const res = await generate(ort, session, pre, {
    maxNewTokens,
    onPrefill: (ms) => post({ type: 'status', id, text: 'Writing…', prefillMs: ms }),
    onToken: (ids) => {
      const now = performance.now();
      if (now - lastSent > 80) { post({ type: 'token', id, text: cleanOutput(detok.decode(ids)), n: ids.length }); lastSent = now; }
      return abort;
    },
  });
  const text = cleanOutput(detok.decode(res.ids));
  post({ type: 'done', id, text, tokens: res.ids.length, prefillMs: res.prefillMs, decodeMs: res.decodeMs, prompt: res.promptTokens, aborted: abort, hitLimit: res.ids.length >= maxNewTokens });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') await load(data);
    else if (data.type === 'ocr') await ocr(data);
    else if (data.type === 'abort') abort = true;
  } catch (e) {
    post({ type: 'error', id: data.id, text: e?.message || String(e) });
  }
};
