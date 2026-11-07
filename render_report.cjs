// Render the report with MarkdownIt and an installed Chromium browser.
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const MarkdownIt = require('markdown-it');

const root = __dirname;
const renderer = new MarkdownIt({ html: true, typographer: false });
renderer.renderer.rules.image = (tokens, index, options, environment, self) => {
  const token = tokens[index];
  const source = token.attrGet('src');
  if (source.startsWith('figures/')) {
    const bytes = fs.readFileSync(path.join(root, source));
    token.attrSet('src', `data:image/svg+xml;base64,${bytes.toString('base64')}`);
  }
  return self.renderToken(tokens, index, options);
};
const body = renderer.render(fs.readFileSync(path.join(root, 'REPORT.md'), 'utf8'));
const html = `<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.5 0.8Bの重み編集による「にゃん」語尾の検証</title>
<style>
body{font-family:"Noto Sans CJK JP",sans-serif;color:#20242a;background:white;
max-width:940px;margin:48px auto;padding:0 28px;font-size:15px;line-height:1.85}
h1{font-size:27px;line-height:1.5;margin:0 0 26px}h2{font-size:21px;margin:38px 0 13px}
h3{font-size:17px;margin:28px 0 10px}p{margin:12px 0}a{color:#185f94}
img{display:block;width:100%;height:auto;margin:20px 0 10px}
table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65;margin:10px 0 24px;
border-top:2px solid #343a40;border-bottom:2px solid #343a40}
th,td{padding:7px 9px;vertical-align:top;overflow-wrap:anywhere}
thead{border-bottom:1px solid #737b82}th{font-weight:500}
code{font-family:"DejaVu Sans Mono","Noto Sans CJK JP",monospace;font-size:.85em;
overflow-wrap:anywhere}pre{padding:14px;background:#f5f6f7;white-space:pre-wrap;line-height:1.6}
li{margin:6px 0}footer{margin-top:40px;padding-top:16px;border-top:1px solid #ddd;font-size:12px}
@page{size:A4;margin:15mm 16mm 17mm;
@bottom-right{content:counter(page) " / " counter(pages);font-family:sans-serif;font-size:8pt;color:#777}}
@media print{
body{margin:0;padding:0;max-width:none;font-size:9.8pt;line-height:1.6}
h1{font-size:20pt;margin-bottom:16pt}h2{font-size:14pt;margin-top:22pt}h3{font-size:11pt}
h1,h2,h3{break-after:avoid}p{orphans:3;widows:3;margin:8pt 0}
table{font-size:8.3pt;margin-bottom:14pt}th,td{padding:4pt 5pt}tr{break-inside:avoid}
thead{display:table-header-group}pre{font-size:8.5pt;break-inside:avoid}
p:has(img){break-inside:avoid;break-after:avoid}p:has(+table){break-after:avoid}
img{margin:12pt 0 4pt}a{text-decoration:none;color:inherit}footer{display:none}
}
</style></head><body>${body}
<footer>実験記録：<a href="results/heldout_outputs.json">全生成結果</a> ／
<a href="results/semantic_review.json">内容の判定理由</a> ／
<a href="results/checkpoint_verification.json">重みの検証</a> ／
<a href="README.md">実行手順</a></footer></body></html>`;
fs.writeFileSync(path.join(root, 'REPORT.html'), html);
const browser = process.env.CHROMIUM_BIN || 'chromium';
const result = spawnSync(browser, [
  '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
  `--print-to-pdf=${path.join(root, 'REPORT.pdf')}`,
  `file://${path.join(root, 'REPORT.html')}`,
], { encoding: 'utf8', timeout: 60000 });
if (result.status !== 0) throw new Error(result.stderr || result.error?.message);
console.log('Wrote REPORT.html and REPORT.pdf');
