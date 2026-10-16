// Falcon-OCR client-side inference core.
// Shared by the browser page (onnxruntime-web) and the Node test (onnxruntime-node).
// Mirrors tiiuae/Falcon-OCR processing_falcon_ocr.py + the ningpp/Falcon-OCR-ONNX
// unified KV-cache graph (falcon_ocr_kv_token.onnx), batch size 1.

export const IDS = {
  pad: 0,
  eos: 11,
  endOfQuery: 263,
  img: 227,
  imgCls: 244,
  imgReg: [245, 246, 247, 248],
  imgEnd: 230,
};

export const PATCH = 16;
export const PATCH_DIM = PATCH * PATCH * 3;
const N_LAYERS = 22;
const N_HEADS = 16;
const HEAD_DIM = 64;

// Pre-tokenized `<instruction>\n<|OCR_PLAIN|>` (the prompt text after <|image|>).
// Tokenizer adds no BOS, and the empty chunk before <|image|> encodes to [].
export const PROMPT_IDS = {
  plain: [46021, 790, 2757, 3463, 1211, 1112, 6883, 537, 709, 257],
  table: [46021, 790, 4336, 3463, 1211, 1112, 6883, 537, 709, 257],
  formula: [46021, 790, 12211, 3463, 1211, 1112, 6883, 537, 709, 257],
};

// Resolution presets: max side for the first resize, and pixel budget for smart_resize.
// Image tokens = pixels / 256; prefill attention memory grows with tokens^2
// (~16 * S^2 * 4 bytes per attention tensor). Downscaling below the reference
// makes the model drift: at 768px a receipt came back as hallucinated JSON.
export const PRESETS = {
  balanced: { maxDim: 1024, maxPixels: 28 * 28 * 640 }, // <= ~1960 image tokens
  high: { maxDim: 1536, maxPixels: 28 * 28 * 1280 }, // reference defaults, <= ~3920 tokens
};

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

// resize_image_if_necessary: clamp sides into [minDim, maxDim], keep aspect.
export function boundedSize(w, h, minDim = 64, maxDim = 1536) {
  if (minDim <= w && w <= maxDim && minDim <= h && h <= maxDim) return [w, h];
  const ar = w / h;
  const vertical = w < h;
  let nw, nh;
  if (w < minDim || h < minDim) {
    if (vertical) { nw = minDim; nh = Math.trunc(nw / ar); }
    else { nh = minDim; nw = Math.trunc(nh * ar); }
  } else if (vertical) { nw = maxDim; nh = Math.trunc(nw / ar); }
  else { nh = maxDim; nw = Math.trunc(nh * ar); }
  if (nw > maxDim) { nw = maxDim; nh = Math.trunc(nw / ar); }
  if (nh > maxDim) { nh = maxDim; nw = Math.trunc(nh * ar); }
  return [nw, nh];
}

// smart_resize: snap to multiples of PATCH within [minPixels, maxPixels].
export function smartSize(w, h, maxPixels, minPixels = 56 * 56) {
  // Python round() is banker's rounding; matters only on exact .5 boundaries.
  const pyRound = (x) => {
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  };
  let hb = pyRound(h / PATCH) * PATCH;
  let wb = pyRound(w / PATCH) * PATCH;
  if (hb * wb > maxPixels) {
    const beta = Math.sqrt((h * w) / maxPixels);
    hb = Math.floor(h / beta / PATCH) * PATCH;
    wb = Math.floor(w / beta / PATCH) * PATCH;
  } else if (hb * wb < minPixels) {
    const beta = Math.sqrt(minPixels / (h * w));
    hb = Math.ceil((h * beta) / PATCH) * PATCH;
    wb = Math.ceil((w * beta) / PATCH) * PATCH;
  }
  return [wb, hb];
}

function bicubic(x) {
  const a = Math.abs(x);
  if (a < 1) return (1.5 * a - 2.5) * a * a + 1;
  if (a < 2) return ((-0.5 * a + 2.5) * a - 4) * a + 2;
  return 0;
}

function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const fscale = Math.max(scale, 1);
  const support = 2 * fscale;
  const bounds = new Int32Array(outSize * 2);
  const weights = [];
  for (let o = 0; o < outSize; o++) {
    const center = (o + 0.5) * scale;
    const lo = Math.max(0, Math.trunc(center - support + 0.5));
    const hi = Math.min(inSize, Math.trunc(center + support + 0.5));
    const w = new Float64Array(hi - lo);
    let sum = 0;
    for (let i = lo; i < hi; i++) sum += (w[i - lo] = bicubic((i - center + 0.5) / fscale));
    if (sum) for (let i = 0; i < w.length; i++) w[i] /= sum;
    bounds[o * 2] = lo; bounds[o * 2 + 1] = hi;
    weights.push(w);
  }
  return { bounds, weights };
}

const clamp8 = (v) => (v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v));

// Pillow-style separable bicubic resize (horizontal pass first, uint8 between passes).
export function resizeRGB(src, sw, sh, dw, dh) {
  if (sw === dw && sh === dh) return src;
  let tmp = src, tw = sw;
  if (dw !== sw) {
    tmp = new Uint8Array(dw * sh * 3);
    const { bounds, weights } = coeffs(sw, dw);
    for (let y = 0; y < sh; y++) {
      const row = y * sw * 3;
      for (let x = 0; x < dw; x++) {
        const lo = bounds[x * 2], hi = bounds[x * 2 + 1], w = weights[x];
        let r = 0, g = 0, b = 0;
        for (let i = lo; i < hi; i++) {
          const k = w[i - lo], p = row + i * 3;
          r += src[p] * k; g += src[p + 1] * k; b += src[p + 2] * k;
        }
        const q = (y * dw + x) * 3;
        tmp[q] = clamp8(r); tmp[q + 1] = clamp8(g); tmp[q + 2] = clamp8(b);
      }
    }
    tw = dw;
  }
  if (dh === sh) return tmp;
  const out = new Uint8Array(dw * dh * 3);
  const { bounds, weights } = coeffs(sh, dh);
  for (let y = 0; y < dh; y++) {
    const lo = bounds[y * 2], hi = bounds[y * 2 + 1], w = weights[y];
    for (let x = 0; x < tw; x++) {
      let r = 0, g = 0, b = 0;
      for (let i = lo; i < hi; i++) {
        const k = w[i - lo], p = (i * tw + x) * 3;
        r += tmp[p] * k; g += tmp[p + 1] * k; b += tmp[p + 2] * k;
      }
      const q = (y * tw + x) * 3;
      out[q] = clamp8(r); out[q + 1] = clamp8(g); out[q + 2] = clamp8(b);
    }
  }
  return out;
}

export function rgbaToRgb(rgba, w, h) {
  const out = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h * 4; i += 4, j += 3) {
    // Composite transparency onto white, like a flattened scan.
    const a = rgba[i + 3] / 255;
    out[j] = Math.round(rgba[i] * a + 255 * (1 - a));
    out[j + 1] = Math.round(rgba[i + 1] * a + 255 * (1 - a));
    out[j + 2] = Math.round(rgba[i + 2] * a + 255 * (1 - a));
  }
  return out;
}

// Full preprocessing: RGB bytes -> model prefill tensors (as plain typed arrays).
export function preprocess(rgb, w, h, { category = 'plain', preset = 'balanced' } = {}) {
  const p = PRESETS[preset] || PRESETS.balanced;
  const [bw, bh] = boundedSize(w, h, 64, p.maxDim);
  let img = resizeRGB(rgb, w, h, bw, bh);
  const [sw, sh] = smartSize(bw, bh, p.maxPixels);
  img = resizeRGB(img, bw, bh, sw, sh);

  const rows = sh / PATCH, cols = sw / PATCH, nImg = rows * cols;
  const prompt = PROMPT_IDS[category] || PROMPT_IDS.plain;
  const ids = [IDS.imgCls, ...IDS.imgReg, ...new Array(nImg).fill(IDS.img), IDS.imgEnd, ...prompt];
  const S = ids.length;
  const firstImg = 1 + IDS.imgReg.length;

  const tokens = new BigInt64Array(S);
  const posT = new BigInt64Array(S);
  const posHw = new Float32Array(S * 2).fill(NaN);
  const patches = new Float32Array(S * PATCH_DIM);
  const noIncrease = new Set([IDS.img, ...IDS.imgReg, IDS.imgEnd]);

  let t = 0;
  for (let i = 0; i < S; i++) {
    tokens[i] = BigInt(ids[i]);
    if (!noIncrease.has(ids[i])) t++;
    posT[i] = BigInt(t - 1);
  }

  // 2D positions: torch.linspace(-lim, lim, n) over the patch grid.
  const xlim = Math.sqrt(cols / rows), ylim = Math.sqrt(rows / cols);
  const lin = (lim, n, k) => (n <= 1 ? -lim : -lim + (2 * lim * k) / (n - 1));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const tok = firstImg + r * cols + c;
      posHw[tok * 2] = lin(ylim, rows, r);
      posHw[tok * 2 + 1] = lin(xlim, cols, c);
      // Patch layout (ph pw c), pixels normalized to [-1, 1].
      let o = tok * PATCH_DIM;
      for (let yy = 0; yy < PATCH; yy++) {
        let s = ((r * PATCH + yy) * sw + c * PATCH) * 3;
        for (let k = 0; k < PATCH * 3; k++) patches[o++] = img[s++] / 127.5 - 1;
      }
    }
  }

  // Causal mask, plus bidirectional attention inside the image block [cls, img_end).
  const mask = new Uint8Array(S * S);
  const imgEndIdx = firstImg + nImg; // index of <|end_of_image|>, which is outside the block
  for (let q = 0; q < S; q++) {
    const lim = q < imgEndIdx ? imgEndIdx - 1 : q;
    mask.fill(1, q * S, q * S + lim + 1);
  }

  return {
    S, rows, cols, width: sw, height: sh, resized: img,
    tokens, posT, posHw, patches, mask,
    lastPosT: Number(posT[S - 1]),
  };
}

// ---------------------------------------------------------------------------
// Detokenizer (byte-level BPE)
// ---------------------------------------------------------------------------

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const m = new Map();
  bs.forEach((b, i) => m.set(String.fromCodePoint(cs[i]), b));
  return m;
}

export class Detokenizer {
  constructor(tokenizerJson) {
    const vocab = tokenizerJson.model.vocab;
    this.tokens = [];
    for (const [tok, id] of Object.entries(vocab)) this.tokens[id] = tok;
    this.special = new Map();
    for (const a of tokenizerJson.added_tokens) {
      this.tokens[a.id] = a.content;
      if (a.special) this.special.set(a.id, a.content);
    }
    this.byteMap = bytesToUnicode();
    this.utf8 = new TextDecoder('utf-8', { fatal: false });
  }

  decode(ids, { skipSpecial = true } = {}) {
    const bytes = [];
    let out = '';
    const flush = () => { if (bytes.length) { out += this.utf8.decode(new Uint8Array(bytes)); bytes.length = 0; } };
    for (const id of ids) {
      if (this.special.has(id)) {
        if (!skipSpecial) { flush(); out += this.special.get(id); }
        continue;
      }
      const tok = this.tokens[id];
      if (tok === undefined) continue;
      for (const ch of tok) {
        const b = this.byteMap.get(ch);
        if (b !== undefined) bytes.push(b);
        else { flush(); out += ch; }
      }
    }
    flush();
    return out;
  }
}

// ---------------------------------------------------------------------------
// Generation loop
// ---------------------------------------------------------------------------

export function cleanOutput(text) {
  return text.replace(/<\|end_of_query\|>|<\|end_of_text\|>/g, '').trim();
}

// ort: onnxruntime-web or onnxruntime-node module. session: InferenceSession.
// onToken(ids) is called after every generated token; return true to abort.
export async function generate(ort, session, pre, { maxNewTokens = 2048, onToken, onPrefill } = {}) {
  const T = (type, data, dims) => new ort.Tensor(type, data, dims);
  const S = pre.S;
  const feeds = {
    tokens: T('int64', pre.tokens, [1, S]),
    image_patches: T('float32', pre.patches, [1, S, PATCH_DIM]),
    pos_t: T('int64', pre.posT, [1, S]),
    pos_hw: T('float32', pre.posHw, [1, S, 2]),
    attention_mask: T('bool', pre.mask, [1, S, S]),
    past_key_values: T('float32', new Float32Array(0), [N_LAYERS, 2, 1, N_HEADS, 0, HEAD_DIM]),
  };

  const t0 = performance.now();
  let out = await session.run(feeds);
  const prefillMs = performance.now() - t0;
  onPrefill?.(prefillMs);

  const zeroPatch = new Float32Array(PATCH_DIM);
  const nanHw = new Float32Array([NaN, NaN]);
  const generated = [];
  let pos = pre.lastPosT;
  let len = S;
  const t1 = performance.now();
  try {
    for (let step = 0; step < maxNewTokens; step++) {
      const next = Number((await out.next_token.getData())[0]);
      generated.push(next);
      if (next === IDS.eos || next === IDS.endOfQuery) break;
      if (onToken?.(generated)) break;
      if (step + 1 >= maxNewTokens) break;

      pos++; len++;
      const past = out.present_key_values;
      const res = await session.run({
        tokens: T('int64', new BigInt64Array([BigInt(next)]), [1, 1]),
        image_patches: T('float32', zeroPatch, [1, 1, PATCH_DIM]),
        pos_t: T('int64', new BigInt64Array([BigInt(pos)]), [1, 1]),
        pos_hw: T('float32', nanHw, [1, 1, 2]),
        attention_mask: T('bool', new Uint8Array(len).fill(1), [1, 1, len]),
        past_key_values: past,
      });
      past.dispose?.();
      out.next_token.dispose?.();
      out = res;
    }
  } finally {
    out.present_key_values?.dispose?.();
  }
  const decodeMs = performance.now() - t1;
  return { ids: generated, prefillMs, decodeMs, promptTokens: S };
}
