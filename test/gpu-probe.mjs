import { chromium } from 'playwright';
const flagSets = [
  ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'],
  ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
];
for (const args of flagSets) {
  const b = await chromium.launch({ args, channel: process.env.CH || undefined });
  const p = await b.newPage();
  await p.goto('http://localhost:8080/index.html');
  const r = await p.evaluate(async () => {
    if (!navigator.gpu) return 'no navigator.gpu';
    const a = await navigator.gpu.requestAdapter();
    if (!a) return 'no adapter';
    return JSON.stringify({ info: a.info, maxBuf: a.limits.maxBufferSize, maxBind: a.limits.maxStorageBufferBindingSize });
  });
  console.log(args.join(' '), '=>', r);
  await b.close();
}
