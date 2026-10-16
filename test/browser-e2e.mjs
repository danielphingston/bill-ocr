// Drives index.html in headless Chromium (WebGPU via SwiftShader) end to end.
// Needs the page served: npm run serve. Usage: node test/browser-e2e.mjs [image] [model-key] [maxTokens]
import { chromium } from 'playwright';

const [img = 'test/images/flux-text.png', model = 'q8', maxTokens = '256'] = process.argv.slice(2);
const browser = await chromium.launch({
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface', '--ignore-certificate-errors'],
});
const page = await (await browser.newContext({ ignoreHTTPSErrors: true })).newPage();
page.on('requestfailed', (r) => console.log('[requestfailed]', r.url().slice(0, 120), r.failure()?.errorText));
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[${m.type()}] ${m.text().slice(0, 300)}`); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
const t0 = Date.now();
const log = (s) => console.log(`${((Date.now() - t0) / 1000).toFixed(1)}s ${s}`);

await page.goto(process.env.URL || 'http://localhost:8080/index.html');
await page.waitForFunction(() => !/Checking/.test(document.querySelector('#gpu').textContent));
log('gpu: ' + (await page.textContent('#gpu')));
await page.selectOption('#model', model);
await page.fill('#max-tokens', maxTokens);
await page.click('#load');
await page.waitForFunction(() => /ready|failed/i.test(document.querySelector('#load-status').textContent), null, { timeout: 15 * 60e3, polling: 1000 });
log('load: ' + (await page.textContent('#load-status')));
await page.setInputFiles('#files', img);
await page.waitForFunction(() => document.querySelector('.result .badge')?.textContent.match(/Done|Error|Stopped/), null, { timeout: 60 * 60e3, polling: 2000 });
log('result: ' + (await page.textContent('.result .badge')) + ' | ' + (await page.textContent('.result .stats')));
const err = await page.$('.result .err');
if (err) log('error: ' + (await err.textContent()));
else {
  await page.click('.result .tabs button[data-tab="raw"]');
  console.log('<<<OCR\n' + (await page.textContent('.result [data-pane="raw"] pre')) + '\nOCR>>>');
  await page.click('.result .tabs button[data-tab="json"]');
  console.log(await page.textContent('.result [data-pane="json"] pre'));
}
await page.screenshot({ path: process.env.SHOT || 'test/screenshot.png', fullPage: true });
await browser.close();
