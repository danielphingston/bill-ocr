import { extractBill } from './extract.js';

// Weights stream straight from Hugging Face (CORS-enabled) and are cached by the browser.
// The source ONNX export is pinned to a commit; the INT8 variant is served from its HF repo.
const HF = 'https://huggingface.co/ningpp/Falcon-OCR-ONNX/resolve/46f25593bc93c19003fbc6b9c968c137476be4c5/';
const HF_Q8 = 'https://huggingface.co/acid1burn/falcon_ocr_kv_token_q8.onnx/resolve/main/';
const MODELS = {
  fp32: { label: 'Full precision · 1.08 GB · Hugging Face', url: [HF + 'falcon_ocr_kv_token.onnx'] },
  q8: { label: '8-bit · 450 MB · Hugging Face', url: [HF_Q8 + 'falcon_ocr_kv_token_q8.onnx'], optional: true },
};
const TOKENIZER = [HF + 'tokenizer.json'];

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const mb = (b) => (b / 1e6).toFixed(b > 1e8 ? 0 : 1) + ' MB';
const secs = (ms) => (ms / 1000).toFixed(1) + 's';

const state = { ready: false, busy: false, queue: [], results: new Map(), nextId: 1, device: null, localFile: null };
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

// ---------------------------------------------------------------------------
// Environment check
// ---------------------------------------------------------------------------

async function detectGpu() {
  const el = $('#gpu');
  if (!('gpu' in navigator)) {
    el.className = 'badge warn'; el.textContent = 'No WebGPU';
    $('#gpu-warning').hidden = false;
    return 'wasm';
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no adapter');
    const info = adapter.info || {};
    const name = [info.vendor, info.architecture || info.device].filter(Boolean).join(' ') || 'GPU';
    el.className = 'badge ok'; el.textContent = `WebGPU · ${name}`;
    el.title = `maxBufferSize ${mb(adapter.limits.maxBufferSize)}, maxStorageBinding ${mb(adapter.limits.maxStorageBufferBindingSize)}`;
    if (adapter.limits.maxStorageBufferBindingSize < 512e6) {
      $('#gpu-small').hidden = false;
    }
    return 'webgpu';
  } catch {
    el.className = 'badge warn'; el.textContent = 'WebGPU unavailable';
    $('#gpu-warning').hidden = false;
    return 'wasm';
  }
}

// Integrated GPUs work but are several times slower than a discrete one. WebGPU has no
// "is integrated" flag, so guess from vendor/architecture (Intel Arc = xe-hpg/xe2-hpg is discrete).
function isIntegrated({ vendor = '', architecture = '', description = '' }) {
  const v = vendor.toLowerCase(), a = architecture.toLowerCase(), d = description.toLowerCase();
  if (v === 'intel') return !/hpg|arc/.test(a + d);
  if (v === 'amd') return /radeon\(tm\) graphics|vega \d+ graphics|680m|780m|880m|890m/.test(d) || /^rdna.*-?apu/.test(a);
  return false;
}

function showAdapter(info) {
  const name = [info.vendor, info.architecture || info.device].filter(Boolean).join(' ') || 'GPU';
  const el = $('#gpu');
  el.textContent = `Running on: ${name}`;
  el.title = info.description || '';
  const integrated = isIntegrated(info);
  el.className = 'badge ' + (integrated ? 'warn' : 'ok');
  $('#igpu-warning').hidden = !integrated;
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

function setLoadUi(loading) {
  $('#load').disabled = loading;
  $('#model').disabled = loading;
  $('#model-file').disabled = loading;
}

function loadModel() {
  const key = $('#model').value;
  // Resolve against the page: the worker would resolve relative URLs against src/.
  const abs = (urls) => urls.map((u) => new URL(u, document.baseURI).href);
  const model = state.localFile || abs(MODELS[key].url);
  setLoadUi(true);
  state.ready = false;
  updateDrop();
  $('#load-progress').hidden = false;
  $('#load-status').textContent = 'Starting…';
  worker.postMessage({ type: 'load', model, tokenizer: abs(TOKENIZER), device: state.device });
}

function onProgress({ label, loaded, total, cached }) {
  const bar = $('#load-bar');
  if (cached) { $('#load-status').textContent = `Loaded ${label} from browser cache`; bar.value = 1; return; }
  bar.value = total ? loaded / total : 0;
  $('#load-status').textContent = `Downloading ${label}: ${mb(loaded)}${total ? ' / ' + mb(total) : ''}`;
}

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------

function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith('image/')) continue;
    const id = state.nextId++;
    const r = { id, file: f, name: f.name || `pasted-${id}.png`, url: URL.createObjectURL(f), status: 'queued', text: '', data: null };
    state.results.set(id, r);
    state.queue.push(id);
    renderCard(r);
  }
  updateExport();
  pump();
}

function pump() {
  if (!state.ready || state.busy || !state.queue.length) return;
  const id = state.queue.shift();
  const r = state.results.get(id);
  state.busy = true;
  r.status = 'running';
  r.started = performance.now();
  renderCard(r);
  worker.postMessage({
    type: 'ocr', id, blob: r.file,
    category: $('#category').value,
    preset: $('#preset').value,
    maxNewTokens: Number($('#max-tokens').value),
  });
}

// Module import failures (e.g. CDN blocked) surface here, not as messages.
worker.onerror = (e) => {
  e.preventDefault?.();
  setLoadUi(false);
  $('#load-status').innerHTML = `<span class="err">Worker failed to start: ${esc(e.message || 'could not load ONNX Runtime from the CDN')}</span>`;
};

worker.onmessage = ({ data: m }) => {
  const r = m.id ? state.results.get(m.id) : null;
  switch (m.type) {
    case 'progress': onProgress(m); break;
    case 'status':
      if (r) { r.note = m.text; if (m.prompt) { r.prompt = m.prompt; r.size = m.size; } if (m.prefillMs) r.prefillMs = m.prefillMs; renderCard(r); }
      else $('#load-status').textContent = m.text;
      break;
    case 'ready':
      state.ready = true;
      if (m.adapter) showAdapter(m.adapter);
      setLoadUi(false);
      $('#load').textContent = 'Reload model';
      $('#load-bar').value = 1;
      $('#load-status').textContent = `Model ready on ${m.device.toUpperCase()} (compiled in ${secs(m.ms)})`;
      updateDrop();
      pump();
      break;
    case 'token':
      if (r) { r.text = m.text; r.tokens = m.n; liveUpdate(r); }
      break;
    case 'done':
      Object.assign(r, { status: 'done', text: m.text, tokens: m.tokens, prefillMs: m.prefillMs, decodeMs: m.decodeMs, prompt: m.prompt, aborted: m.aborted, hitLimit: m.hitLimit });
      r.data = extractBill(m.text);
      state.busy = false;
      renderCard(r);
      updateExport();
      pump();
      break;
    case 'error':
      if (r) {
        r.status = 'error'; r.note = m.text; state.busy = false; renderCard(r); pump();
      } else {
        setLoadUi(false);
        $('#load-status').innerHTML = `<span class="err">Load failed: ${esc(m.text)}</span>`;
      }
      break;
  }
};

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function fieldRows(d) {
  const rows = [
    ['Vendor', d.vendor?.name], ['Address', d.vendor?.address], ['Phone', d.vendor?.phone], ['Email', d.vendor?.email],
    ['Tax ID', d.vendor?.tax_id], ['Number', d.document?.number], ['Date', d.document?.date], ['Time', d.document?.time],
    ['Currency', d.currency], ['Subtotal', d.totals?.subtotal], ['Discount', d.totals?.discount], ['Shipping', d.totals?.shipping],
    ['Tax', d.totals?.tax], ['Tip', d.totals?.tip], ['Total', d.totals?.total],
    ['Payment', d.payment && [d.payment.method, d.payment.card_last4 && '•••• ' + d.payment.card_last4].filter(Boolean).join(' ')],
  ].filter(([, v]) => v !== undefined && v !== null && v !== '');
  if (!rows.length) return '<p class="muted">No bill fields recognised. Check the Raw OCR tab.</p>';
  return `<table class="kv">${rows.map(([k, v]) => `<tr><th>${k}</th><td${k === 'Total' ? ' class="total"' : ''}>${esc(v)}</td></tr>`).join('')}</table>`;
}

function checksHtml(c) {
  if (!c) return '';
  const item = (ok, yes, no) => `<li class="${ok ? 'ok' : 'warn'}">${ok ? '✓' : '⚠'} ${ok ? yes : no}</li>`;
  return `<ul class="checks">${'totals_consistent' in c ? item(c.totals_consistent, 'Subtotal + tax = total', 'Subtotal + tax ≠ total — double-check the numbers') : ''}${'items_match_subtotal' in c ? item(c.items_match_subtotal, 'Line items add up', 'Line items don’t add up to the subtotal') : ''}</ul>`;
}

function itemsHtml(items) {
  if (!items?.length) return '<p class="muted">No line items found.</p>';
  return `<table class="items"><thead><tr><th>Description</th><th>Qty</th><th>Unit</th><th>Amount</th></tr></thead><tbody>${items
    .map((i) => `<tr><td>${esc(i.description)}</td><td class="num">${esc(i.quantity ?? '')}</td><td class="num">${esc(i.unit_price ?? '')}</td><td class="num">${esc(i.amount ?? '')}</td></tr>`)
    .join('')}</tbody></table>`;
}

function statsLine(r) {
  const parts = [];
  if (r.size) parts.push(`${r.size[0]}×${r.size[1]}px`);
  if (r.prompt) parts.push(`${r.prompt} input tokens`);
  if (r.prefillMs) parts.push(`read ${secs(r.prefillMs)}`);
  if (r.tokens && r.decodeMs) parts.push(`${r.tokens} tokens in ${secs(r.decodeMs)} (${(r.tokens / (r.decodeMs / 1000)).toFixed(1)}/s)`);
  else if (r.tokens) parts.push(`${r.tokens} tokens…`);
  return parts.join(' · ');
}

function renderCard(r) {
  let card = document.getElementById('r' + r.id);
  if (!card) {
    card = document.createElement('article');
    card.id = 'r' + r.id;
    card.className = 'result';
    $('#results').prepend(card);
  }
  const badge = { queued: 'Queued', running: 'Reading…', done: r.aborted ? 'Stopped' : 'Done', error: 'Error' }[r.status];
  const tab = card.dataset.tab || 'fields';
  const body =
    r.status === 'done'
      ? `<nav class="tabs">${['fields', 'items', 'raw', 'json'].map((t) => `<button data-tab="${t}" class="${t === tab ? 'on' : ''}">${{ fields: 'Fields', items: `Line items (${r.data.line_items?.length || 0})`, raw: 'Raw OCR', json: 'JSON' }[t]}</button>`).join('')}</nav>
         <div class="pane" data-pane="fields" ${tab === 'fields' ? '' : 'hidden'}>${fieldRows(r.data)}${checksHtml(r.data.checks)}</div>
         <div class="pane" data-pane="items" ${tab === 'items' ? '' : 'hidden'}>${itemsHtml(r.data.line_items)}</div>
         <div class="pane" data-pane="raw" ${tab === 'raw' ? '' : 'hidden'}><pre>${esc(r.text)}</pre></div>
         <div class="pane" data-pane="json" ${tab === 'json' ? '' : 'hidden'}><pre>${esc(JSON.stringify(r.data, null, 2))}</pre></div>
         <div class="row">
           <button class="small" data-act="copy">Copy JSON</button>
           <button class="small" data-act="download">Download JSON</button>
           <button class="small ghost" data-act="rerun">Run again</button>
         </div>`
      : r.status === 'running'
        ? `<pre class="live">${esc(r.text) || '<span class="muted">' + esc(r.note || 'Preparing image…') + '</span>'}</pre><div class="row"><button class="small ghost" data-act="stop">Stop</button></div>`
        : r.status === 'error'
          ? `<p class="err">${esc(r.note)}</p><p class="muted">Large images need more GPU memory — try the Balanced resolution, or crop the image.</p><div class="row"><button class="small ghost" data-act="rerun">Try again</button></div>`
          : `<p class="muted">${state.ready ? 'Waiting for the previous image…' : 'Waiting — load the model in step 1.'}</p>`;
  const warn = r.status === 'done' && r.hitLimit ? `<p class="warn-inline">⚠ Hit the max-token limit — the text may be cut off. Raise “Max output tokens” and run again.</p>` : '';
  card.innerHTML = `
    <img src="${r.url}" alt="" class="thumb">
    <div class="body">
      <header><strong title="${esc(r.name)}">${esc(r.name)}</strong><span class="badge ${r.status}">${badge}</span></header>
      <div class="stats">${statsLine(r)}</div>
      ${warn}${body}
    </div>`;
}

function liveUpdate(r) {
  const card = document.getElementById('r' + r.id);
  const pre = card && $('.live', card);
  if (!pre) return renderCard(r);
  pre.textContent = r.text;
  pre.scrollTop = pre.scrollHeight;
  $('.stats', card).textContent = statsLine(r);
}

$('#results').addEventListener('click', (e) => {
  const card = e.target.closest('.result');
  if (!card) return;
  const r = state.results.get(Number(card.id.slice(1)));
  const tab = e.target.dataset.tab;
  if (tab) { card.dataset.tab = tab; renderCard(r); return; }
  const act = e.target.dataset.act;
  if (act === 'copy') navigator.clipboard.writeText(JSON.stringify(r.data, null, 2)).then(() => { e.target.textContent = 'Copied ✓'; });
  if (act === 'download') download(`${r.name.replace(/\.[^.]+$/, '')}.json`, JSON.stringify({ file: r.name, ...r.data, raw_text: r.text }, null, 2), 'application/json');
  if (act === 'stop') worker.postMessage({ type: 'abort' });
  if (act === 'rerun') { r.status = 'queued'; r.text = ''; r.tokens = 0; r.decodeMs = 0; state.queue.push(r.id); renderCard(r); pump(); }
});

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const done = () => [...state.results.values()].filter((r) => r.status === 'done');

function updateExport() {
  const n = done().length;
  $('#export').hidden = n === 0;
  $('#export-count').textContent = `${n} bill${n === 1 ? '' : 's'}`;
}

$('#export-json').onclick = () => download('bills.json', JSON.stringify(done().map((r) => ({ file: r.name, ...r.data, raw_text: r.text })), null, 2), 'application/json');
$('#export-csv').onclick = () => {
  const cols = ['file', 'vendor', 'number', 'date', 'currency', 'subtotal', 'tax', 'total', 'payment', 'items'];
  const q = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const rows = done().map((r) => {
    const d = r.data;
    return [r.name, d.vendor?.name, d.document?.number, d.document?.date, d.currency, d.totals?.subtotal, d.totals?.tax, d.totals?.total, d.payment?.method, d.line_items?.length || 0].map(q).join(',');
  });
  download('bills.csv', [cols.join(','), ...rows].join('\n'), 'text/csv');
};

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

function updateDrop() {
  $('#drop').classList.toggle('disabled', !state.ready);
  $('#drop-hint').textContent = state.ready ? 'PNG, JPG or WebP · several at once is fine' : 'You can add images now — they’ll run once the model is loaded';
}

const drop = $('#drop');
drop.onclick = () => $('#files').click();
drop.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#files').click(); } };
$('#files').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); addFiles(e.dataTransfer.files); };
window.addEventListener('paste', (e) => addFiles([...e.clipboardData.items].filter((i) => i.kind === 'file').map((i) => i.getAsFile())));

$('#load').onclick = loadModel;
$('#model').onchange = () => { state.localFile = null; $('#model-file-name').textContent = ''; };
$('#model-file').onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  state.localFile = f;
  $('#model-file-name').textContent = `Using ${f.name} (${mb(f.size)})`;
};

for (const [key, m] of Object.entries(MODELS)) {
  const o = document.createElement('option');
  o.value = key;
  o.textContent = m.label;
  $('#model').append(o);
  if (m.optional) {
    // Probe without downloading; drop the option when the file isn't hosted next to the page.
    fetch(new URL(m.url[0], document.baseURI), { method: 'HEAD' })
      .then((r) => { if (!r.ok) throw 0; })
      .catch(() => o.remove());
  }
}

state.device = await detectGpu();
updateDrop();
