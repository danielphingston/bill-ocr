// Extractor sanity checks on representative Falcon-OCR outputs.
import assert from 'node:assert/strict';
import { extractBill } from '../src/extract.js';

const plain = `\`\`\`
GREEN LEAF MARKET
221 Harbor Road, Springfield
Tel: (555) 013-4477

Invoice #: INV-20931
Date: 2026-09-14 Time: 14:32

Item                     Qty      Amount
---------------------------
Organic Apples            2        6.98
Whole Milk 1L             1        2.49
Sourdough Bread           1        4.75
Coffee Beans              1       12.90
Olive Oil 500ml           1        8.30
---------------------------
Subtotal                          35.42
Tax (8%)                           2.83
TOTAL                            $38.25
Paid: VISA **** 4821
Thank you for shopping!
\`\`\``;

const html = `<table><tr><th>Description</th><th>Qty</th><th>Unit Price</th><th>Amount</th></tr>
<tr><td>Consulting services</td><td>10</td><td>120,00</td><td>1.200,00</td></tr>
<tr><td>Hosting (annual)</td><td>1</td><td>300,00</td><td>300,00</td></tr>
<tr><td>Subtotal</td><td></td><td></td><td>1.500,00</td></tr></table>
MwSt 19% 285,00 €
Gesamtbetrag 1.785,00 €
Rechnungsdatum: 03.10.2026`;

const a = extractBill(plain);
console.log(JSON.stringify(a, null, 1));
assert.equal(a.vendor.name, 'GREEN LEAF MARKET');
assert.equal(a.document.number, 'INV-20931');
assert.equal(a.document.date, '2026-09-14');
assert.equal(a.currency, 'USD');
assert.equal(a.line_items.length, 5);
assert.deepEqual(a.totals, { subtotal: 35.42, tax: 2.83, total: 38.25 });
assert.equal(a.payment.card_last4, '4821');
assert.equal(a.checks.totals_consistent, true);
assert.equal(a.checks.items_match_subtotal, true);

const b = extractBill(html);
console.log(JSON.stringify(b, null, 1));
assert.equal(b.line_items.length, 2);
assert.deepEqual(b.line_items[0], { description: 'Consulting services', quantity: 10, unit_price: 120, amount: 1200 });
assert.equal(b.totals.subtotal, 1500);
assert.equal(b.currency, 'EUR');
console.log('extract tests passed');
