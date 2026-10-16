// Rule-based bill/receipt/invoice field extraction from Falcon-OCR text output.
// Falcon-OCR emits plain text, Markdown, or HTML <table> markup; all are handled.

const CURRENCY_SYMBOLS = { '$': 'USD', '€': 'EUR', '£': 'GBP', '₹': 'INR', '¥': 'JPY', '₩': 'KRW', '₺': 'TRY', '₽': 'RUB', '₱': 'PHP', '฿': 'THB', 'R$': 'BRL', 'A$': 'AUD', 'C$': 'CAD', 'S$': 'SGD', 'HK$': 'HKD' };
const CURRENCY_CODES = /\b(USD|EUR|GBP|INR|JPY|CNY|RMB|AUD|CAD|CHF|SGD|HKD|AED|SAR|ZAR|NZD|SEK|NOK|DKK|PLN|MXN|BRL|KRW|THB|MYR|IDR|PHP|VND|TRY|RUB)\b/;

// Money: 1,234.56 | 1.234,56 | 1234,56 | 12.5 | -3.00 ; optional currency around it.
const AMOUNT_RE = /(?:[A-Z]{3}\s?|[A-Z]?\$|[€£₹¥₩₺₽₱฿])?\s?-?\(?\d{1,3}(?:[ ,. ]\d{3})*(?:[.,]\d{1,2})?\)?(?!\d)|(?:[A-Z]?\$|[€£₹¥₩₺₽₱฿])\s?-?\d+(?:[.,]\d{1,2})?/g;

export function parseAmount(raw) {
  if (raw == null) return null;
  let s = String(raw).replace(/[^\d,.\-()]/g, '');
  const neg = /^\(.*\)$/.test(s) || s.startsWith('-');
  s = s.replace(/[()-]/g, '');
  if (!s) return null;
  const lastDot = s.lastIndexOf('.'), lastComma = s.lastIndexOf(',');
  const dec = Math.max(lastDot, lastComma);
  // A separator followed by exactly 1-2 digits at the end is the decimal mark.
  if (dec >= 0 && s.length - dec - 1 <= 2 && s.length - dec - 1 > 0) {
    s = s.slice(0, dec).replace(/[.,]/g, '') + '.' + s.slice(dec + 1);
  } else {
    s = s.replace(/[.,]/g, '');
  }
  const n = Number(s);
  if (!Number.isFinite(n)) return null;
  return neg ? -n : n;
}

function amountsIn(line) {
  const out = [];
  for (const m of line.matchAll(AMOUNT_RE)) {
    const txt = m[0].trim();
    if (!/\d/.test(txt)) continue;
    // Skip things that look like dates, times, phone numbers, or percentages.
    const after = line.slice(m.index + m[0].length, m.index + m[0].length + 1);
    if (after === '%' || after === ':' || after === '/') continue;
    const before = line.slice(Math.max(0, m.index - 1), m.index);
    if (before === '/' || before === ':') continue;
    const v = parseAmount(txt);
    if (v !== null) out.push({ text: txt, value: v, hasDecimals: /[.,]\d{2}\)?$/.test(txt), hasSymbol: /[$€£₹¥₩₺₽₱฿]|[A-Z]{3}/.test(txt) });
  }
  return out;
}

const lastMoney = (line) => {
  const a = amountsIn(line).filter((x) => x.hasDecimals || x.hasSymbol);
  return a.length ? a[a.length - 1].value : null;
};

// ---------------------------------------------------------------------------
// Text normalization: HTML tables / Markdown -> lines + table rows
// ---------------------------------------------------------------------------

function decodeEntities(s) {
  return s.replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'");
}

function htmlTables(text) {
  const tables = [];
  for (const t of text.matchAll(/<table[\s\S]*?<\/table>/gi)) {
    const rows = [];
    for (const r of t[0].matchAll(/<tr[\s\S]*?<\/tr>/gi)) {
      const cells = [...r[0].matchAll(/<t[hd][^>]*>([\s\S]*?)<\/t[hd]>/gi)].map((c) => decodeEntities(c[1].replace(/<[^>]+>/g, ' ')).replace(/\s+/g, ' ').trim());
      if (cells.length) rows.push(cells);
    }
    if (rows.length) tables.push(rows);
  }
  return tables;
}

function markdownTables(lines) {
  const tables = [];
  let cur = null;
  for (const l of lines) {
    if (/^\s*\|.*\|\s*$/.test(l)) {
      if (/^\s*\|[\s:|-]+\|\s*$/.test(l)) continue; // separator row
      const cells = l.trim().slice(1, -1).split('|').map((c) => c.trim());
      (cur ||= []).push(cells);
    } else if (cur) { tables.push(cur); cur = null; }
  }
  if (cur) tables.push(cur);
  return tables;
}

export function toLines(text) {
  const flat = decodeEntities(
    text
      .replace(/<\/(tr|p|div|h\d|li)>/gi, '\n')
      .replace(/<br\s*\/?>/gi, '\n')
      .replace(/<\/t[hd]>/gi, '  ')
      .replace(/<[^>]+>/g, ''),
  );
  return flat
    .split(/\n/)
    .map((l) => l.replace(/^#{1,6}\s+/, '').replace(/\*\*(\S[^*]*?)\*\*/g, '$1').replace(/\s+$/, ''))
    .filter((l) => l.trim());
}

// ---------------------------------------------------------------------------
// Field rules
// ---------------------------------------------------------------------------

const KEYS = {
  grandTotal: /\b(grand\s*total|total\s*(amount\s*)?due|amount\s*due|balance\s*due|total\s*payable|net\s*payable|amount\s*payable|total\s*to\s*pay|invoice\s*total|total\s*incl|total\b(?!\s*(items?|qty|quantity)))/i,
  subtotal: /\b(sub\s*-?\s*total|net\s*amount|total\s*excl|amount\s*before\s*tax|taxable\s*amount)/i,
  tax: /\b(tax|vat|gst|hst|pst|cgst|sgst|igst|sales\s*tax|mwst|tva|iva)\b/i,
  tip: /\b(tip|gratuity|service\s*charge)\b/i,
  discount: /\b(discount|savings|coupon|promo)\b/i,
  shipping: /\b(shipping|delivery|freight|postage)\b/i,
  paid: /\b(cash|change|tendered|paid|visa|mastercard|master\s*card|amex|american\s*express|debit|credit|card|upi|paypal|apple\s*pay|google\s*pay)\b/i,
};
const NON_ITEM = new RegExp([KEYS.grandTotal, KEYS.subtotal, KEYS.tax, KEYS.tip, KEYS.discount, KEYS.shipping, KEYS.paid].map((r) => r.source).join('|') + '|\\b(invoice|receipt|date|time|tel|phone|fax|email|www\\.|http|order|table|server|cashier|balance|thank)', 'i');

const MONTHS = 'jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?';
const DATE_RES = [
  /\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b/, // ISO
  /\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b/, // D/M/Y or M/D/Y
  new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?[\\s-]+(${MONTHS})[a-z]*\\.?[\\s,-]+(\\d{2,4})\\b`, 'i'),
  new RegExp(`\\b(${MONTHS})[a-z]*\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?,?\\s+(\\d{2,4})\\b`, 'i'),
];

function normDate(m, idx) {
  const mon = (s) => 'janfebmaraprmayjunjulaugsepoctnovdec'.indexOf(s.slice(0, 3).toLowerCase()) / 3 + 1;
  const yy = (y) => (y.length === 2 ? '20' + y : y);
  const p = (n) => String(n).padStart(2, '0');
  let y, mo, d;
  if (idx === 0) [y, mo, d] = [m[1], +m[2], +m[3]];
  else if (idx === 1) {
    const a = +m[1], b = +m[2];
    y = yy(m[3]);
    // Ambiguous D/M vs M/D: prefer D/M when first part > 12, else M/D.
    if (a > 12) [d, mo] = [a, b]; else [mo, d] = [a, b];
  } else if (idx === 2) [d, mo, y] = [+m[1], mon(m[2]), yy(m[3])];
  else [mo, d, y] = [mon(m[1]), +m[2], yy(m[3])];
  if (!(mo >= 1 && mo <= 12 && d >= 1 && d <= 31)) return null;
  return `${y}-${p(mo)}-${p(d)}`;
}

function findDate(lines) {
  const ordered = [...lines].sort((a, b) => /date|dated|issued/i.test(b) - /date|dated|issued/i.test(a));
  for (const l of ordered) {
    for (let i = 0; i < DATE_RES.length; i++) {
      const m = l.match(DATE_RES[i]);
      if (m) { const v = normDate(m, i); if (v) return { value: v, raw: m[0] }; }
    }
  }
  return null;
}

const firstMatch = (lines, re, g = 1) => { for (const l of lines) { const m = l.match(re); if (m) return m[g].trim(); } return null; };

function findCurrency(text) {
  for (const sym of Object.keys(CURRENCY_SYMBOLS).sort((a, b) => b.length - a.length)) {
    if (text.includes(sym)) return CURRENCY_SYMBOLS[sym];
  }
  const m = text.match(CURRENCY_CODES);
  return m ? (m[1] === 'RMB' ? 'CNY' : m[1]) : null;
}

function amountFor(lines, re, exclude) {
  let best = null;
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (!re.test(l) || (exclude && exclude.test(l))) continue;
    let v = lastMoney(l);
    // Label and amount on separate lines.
    if (v === null && i + 1 < lines.length && !/[a-z]{3,}/i.test(lines[i + 1].replace(/[A-Z]{3}/, ''))) v = lastMoney(lines[i + 1]);
    if (v !== null) best = { value: v, line: l.trim() };
  }
  return best; // last occurrence wins: totals sit at the bottom
}

// ---------------------------------------------------------------------------
// Line items
// ---------------------------------------------------------------------------

function itemsFromTables(tables) {
  const items = [];
  for (const rows of tables) {
    let header = rows[0].map((c) => c.toLowerCase());
    const hasHeader = header.some((c) => /item|desc|product|article|particular|name|qty|quantity|price|amount|total/.test(c));
    const col = (re) => (hasHeader ? header.findIndex((c) => re.test(c)) : -1);
    const ci = {
      desc: col(/item|desc|product|article|particular|name|service/),
      qty: col(/qty|quantity|units?|pcs|no\./),
      unit: col(/unit\s*price|rate|price|each|u\.?p/),
      amount: col(/amount|total|line|sum|value|net/),
    };
    for (const r of hasHeader ? rows.slice(1) : rows) {
      const joined = r.join(' ');
      if (!r.some((c) => /\d/.test(c)) || NON_ITEM.test(joined)) continue;
      const nums = r.map((c) => (/\d/.test(c) && !/[a-z]{3,}/i.test(c) ? parseAmount(c) : null));
      const desc = ci.desc >= 0 ? r[ci.desc] : r.find((c, i) => nums[i] === null) || r[0];
      if (!desc || !/[a-z]/i.test(desc)) continue;
      const pick = (i) => (i >= 0 && i < r.length ? nums[i] : null);
      const numericIdx = nums.map((n, i) => (n !== null ? i : -1)).filter((i) => i >= 0);
      const amount = pick(ci.amount) ?? (numericIdx.length ? nums[numericIdx[numericIdx.length - 1]] : null);
      const qty = pick(ci.qty) ?? (numericIdx.length >= 2 && Number.isInteger(nums[numericIdx[0]]) ? nums[numericIdx[0]] : null);
      const unit = pick(ci.unit) ?? (numericIdx.length >= 3 ? nums[numericIdx[numericIdx.length - 2]] : null);
      items.push(clean({ description: desc, quantity: qty, unit_price: unit, amount }));
    }
  }
  return items;
}

// Some OCR table outputs keep only an outer <table> tag and flatten all rows
// into one string. Recover serial-numbered rows from that representation.
function itemsFromFlatTable(text) {
  const flat = decodeEntities(text.replace(/<[^>]+>/g, ' ')).replace(/\s+/g, ' ').trim();
  const header = /\bsl\.?\s*description\s*qty\s*rate\s*amount\b/i.exec(flat);
  if (!header) return [];
  const body = flat.slice(header.index + header[0].length).split(/\b(?:subtotal|payment\s+instructions)\b/i)[0]
    .replace(/([a-z])(\d+(?:[.,]\d+)?)\s*(?=\$)/gi, '$1 $2 ')
    .replace(/(\d+\.\d{2})(?=\d{1,3}\s+[a-z])/gi, '$1 ');
  const row = /(?:^|\s)\d{1,3}\s+([a-z][\s\S]*?)\s+(\d+(?:[.,]\d+)?)\s*\$?\s*([\d,]+\.\d{2})\s*\$?\s*([\d,]+\.\d{2})(?=\s+\d{1,3}\s+[a-z]|\s*$)/gi;
  const items = [];
  for (const m of body.matchAll(row)) {
    const description = m[1].trim();
    if (!description || NON_ITEM.test(description)) continue;
    items.push(clean({ description, quantity: parseAmount(m[2]), unit_price: parseAmount(m[3]), amount: parseAmount(m[4]) }));
  }
  return items;
}

function itemsFromModelJson(root) {
  const arrays = [];
  const listKey = /^(?:lineitems?|items?|menu|products?|details|entries|rows|orderlines)$/i;
  const visit = (value, key = '') => {
    if (Array.isArray(value)) {
      if (listKey.test(key) || (!key && value.every((x) => x && typeof x === 'object' && !Array.isArray(x)))) arrays.push(value);
      else for (const x of value) visit(x);
    } else if (value && typeof value === 'object') {
      for (const [k, v] of Object.entries(value)) visit(v, k.replace(/[^a-z]/gi, ''));
    }
  };
  visit(root);

  for (const rows of arrays) {
    const items = rows.flatMap((row) => {
      if (!row || typeof row !== 'object' || Array.isArray(row)) return [];
      const fields = Object.fromEntries(Object.entries(row).map(([k, v]) => [k.toLowerCase().replace(/[^a-z0-9]/g, ''), v]));
      const get = (...keys) => keys.map((k) => fields[k]).find((v) => v !== undefined && v !== null && v !== '');
      const rawDesc = get('description', 'itemname', 'productname', 'name', 'product', 'item', 'nm', 'desc', 'particular', 'particulars');
      if (rawDesc === undefined || typeof rawDesc === 'object') return [];
      const description = String(rawDesc).trim();
      if (!/[a-z]{2,}/i.test(description)) return [];
      const quantity = parseAmount(get('quantity', 'qty', 'count', 'cnt', 'units'));
      let unit = parseAmount(get('unitprice', 'rate', 'priceperunit', 'each', 'unitcost'));
      let amount = parseAmount(get('amount', 'linetotal', 'lineamount', 'total', 'sum', 'value', 'price'));
      if (amount === null && unit !== null && quantity !== null) amount = Math.round(unit * quantity * 100) / 100;
      if (unit === null && amount !== null && quantity !== null && quantity !== 0) unit = Math.round((amount / quantity) * 100) / 100;
      return [clean({ description, quantity, unit_price: unit, amount })];
    });
    if (items.length) return items;
  }
  return [];
}

function itemsFromLines(lines) {
  const items = [];
  const hasSerialColumn = lines.some((l) => /(?:^|\s)(?:sl\.?|s\.?\s*no\.?|serial\s*(?:no\.?|#)?|item\s*(?:no\.?|#))(?=.*\bdesc(?:ription)?\b)/i.test(l));
  for (const l of lines) {
    if (NON_ITEM.test(l) || /^[-=_*.\s]+$/.test(l)) continue;
    const amts = amountsIn(l).filter((a) => a.hasDecimals);
    if (!amts.length) continue;
    const last = amts[amts.length - 1];
    const head = l.slice(0, l.lastIndexOf(last.text)).trim();
    if (!/[a-z]{2,}/i.test(head)) continue;
    // "<desc> <qty> [x] [unit] <amount>" or "<qty> x <desc> <amount>".
    let m = head.match(/^(.*?[a-z].*?)\s+(\d+(?:[.,]\d+)?)\s*(?:x|@|×)?\s*([$€£₹¥]?\s?\d+[.,]\d{2})?$/i);
    let desc = head, qty = null, unit = null;
    if (m && m[2] && !/^\d{4,}$/.test(m[2])) { desc = m[1]; qty = parseAmount(m[2]); unit = m[3] ? parseAmount(m[3]) : null; }
    else if ((m = head.match(/^(\d+)\s*(?:x|×|@)\s*(.+)$/i))) { qty = +m[1]; desc = m[2]; }
    if (hasSerialColumn) desc = desc.replace(/^\s*\d+[.)]?\s+(?=[a-z])/i, '');
    desc = desc.replace(/[|:]+$/, '').replace(/\s{2,}/g, ' ').trim();
    if (desc.length < 2) continue;
    items.push(clean({ description: desc, quantity: qty, unit_price: unit, amount: last.value }));
  }
  return items;
}

function clean(o) {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== null && v !== undefined && v !== ''));
}

// ---------------------------------------------------------------------------

// Falcon-OCR sometimes answers receipts with a JSON object (CORD-style keys).
function embeddedJson(text) {
  const m = text.match(/```(?:json)?\s*([\s\S]*?)(?:```|$)/) || [null, text];
  const body = m[1].trim();
  if (!/^[\[{]/.test(body)) return null;
  try { return JSON.parse(body); } catch { return null; }
}

export function extractBill(rawText) {
  const modelJson = embeddedJson(rawText);
  const text = rawText.replace(/^```[a-z]*\s*$/gim, '');
  const lines = toLines(text);
  const tables = [...htmlTables(text), ...markdownTables(text.split('\n'))];
  const all = lines.join('\n');

  const header = lines.slice(0, 6).filter((l) => !DATE_RES.some((r) => r.test(l)) && !/\b(tel|phone|fax|invoice|receipt\s*(no|#))\b/i.test(l));
  const vendor = header.find((l) => /[a-z]{2,}/i.test(l) && !/^\s*[\d\W]+\s*$/.test(l) && l.trim().length <= 60)?.trim() || null;

  const address = lines.slice(0, 10).find((l) => /\d+\s+\w+.*\b(st|street|rd|road|ave|avenue|blvd|lane|ln|dr|drive|way|plaza|suite|floor|hwy|highway|strasse|straße|via|rue)\b/i.test(l))?.trim() || null;
  const phone = firstMatch(lines, /(?:tel|phone|ph|mob|mobile|t)\s*[.:#]?\s*([+()\d][\d\s().-]{6,}\d)/i) || firstMatch(lines, /(\+?\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4})/);
  const email = firstMatch(lines, /([\w.+-]+@[\w-]+\.[\w.-]+)/);
  const website = firstMatch(lines, /\b((?:https?:\/\/)?(?:www\.)[\w-]+\.[\w.\/-]+)/i);
  const taxId = firstMatch(lines, /\b(?:vat|gst|tax|tin|abn|ein|gstin|siret|ust-?id)\s*(?:reg\.?\s*)?(?:no\.?|number|id|#)?\s*[:.]?\s*([A-Z0-9][A-Z0-9 -]{5,})/i);
  const docNumber = firstMatch(lines, /\b(?:invoice|inv|receipt|bill|order|transaction|txn|trans|ref(?:erence)?|doc(?:ument)?|ticket|check|chk)\s*(?:no\.?|number|num|#|id)?\s*[:#.]?\s*([A-Z]{0,5}-?\d[\w/-]*)/i);
  const date = findDate(lines);
  const time = firstMatch(lines, /\b((?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\s*(?:[ap]\.?m\.?)?)/i);

  const subtotal = amountFor(lines, KEYS.subtotal);
  const tax = amountFor(lines, KEYS.tax, /\b(incl|excl|before|total\s*excl)/i);
  const tip = amountFor(lines, KEYS.tip);
  const discount = amountFor(lines, KEYS.discount);
  const shipping = amountFor(lines, KEYS.shipping);
  let total = amountFor(lines, KEYS.grandTotal, /\bsub\s*-?\s*total|total\s*(tax|vat|discount|savings|items?|qty)/i);
  if (!total) {
    // Fallback: largest money value on the page.
    const vals = lines.flatMap((l) => amountsIn(l).filter((a) => a.hasDecimals).map((a) => a.value));
    if (vals.length) total = { value: Math.max(...vals), line: '(largest amount)' };
  }

  const payLine = lines.find((l) => /\b(visa|mastercard|master\s*card|amex|american\s*express|discover|maestro|debit|credit|cash|upi|paypal|apple\s*pay|google\s*pay)\b/i.test(l));
  const payment = payLine
    ? clean({
        method: (payLine.match(/\b(visa|mastercard|master\s*card|amex|american\s*express|discover|maestro|debit|credit|cash|upi|paypal|apple\s*pay|google\s*pay)\b/i) || [])[1]?.toUpperCase().replace(/\s+/g, ' '),
        card_last4: (payLine.match(/(?:\*{2,}|x{2,}|#{2,}|\.{3,})\s*(\d{4})\b/i) || [])[1] || null,
      })
    : null;

  let items = itemsFromModelJson(modelJson);
  if (!items.length) items = itemsFromTables(tables);
  if (!items.length) items = itemsFromFlatTable(text);
  if (!items.length) items = itemsFromLines(lines);
  const itemsSum = items.reduce((s, i) => s + (i.amount || 0), 0);

  const checks = {};
  if (subtotal && total) {
    const expect = subtotal.value + (tax?.value || 0) + (tip?.value || 0) + (shipping?.value || 0) - Math.abs(discount?.value || 0);
    checks.totals_consistent = Math.abs(expect - total.value) < 0.02;
  }
  if (items.length && (subtotal || total)) {
    checks.items_match_subtotal = Math.abs(itemsSum - (subtotal || total).value) < 0.02;
  }

  return clean({
    vendor: clean({ name: vendor, address, phone, email, website, tax_id: taxId }),
    document: clean({ number: docNumber, date: date?.value, date_raw: date?.raw, time }),
    currency: findCurrency(all),
    line_items: items,
    totals: clean({
      subtotal: subtotal?.value,
      discount: discount?.value,
      shipping: shipping?.value,
      tax: tax?.value,
      tip: tip?.value,
      total: total?.value,
    }),
    payment,
    checks: Object.keys(checks).length ? checks : null,
    model_json: modelJson,
  });
}
