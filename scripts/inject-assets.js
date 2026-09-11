hexo.extend.injector.register(
  'head_end',
  '<link rel="stylesheet" href="/blog/assets/site.css">'
);

hexo.extend.injector.register(
  'head_end',
  '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.18.4/dist/katex.min.css">'
);

hexo.extend.injector.register(
  'post',
  `<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

// Hexo renders fenced code with unknown languages (mermaid) as
// <figure class="highlight plaintext"><table>...<span class="line">...</span><br>...
// so detect diagrams by their first line and rebuild the source with newlines.
const MERMAID_HEAD = /^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|pie|mindmap|timeline|gitGraph|quadrantChart|requirementDiagram|C4Context|sankey|xychart|block|packet|architecture|radar)\\b/i;

document.querySelectorAll('figure.highlight').forEach((fig) => {
  const pre = fig.querySelector('td.code pre');
  if (!pre) return;
  const lines = Array.from(pre.querySelectorAll('span.line'), (s) => s.textContent);
  if (!lines.length || !MERMAID_HEAD.test(lines[0])) return;
  const holder = document.createElement('div');
  holder.className = 'mermaid';
  holder.textContent = lines.join('\\n').trim();
  fig.replaceWith(holder);
});

mermaid.initialize({ startOnLoad: false, theme: 'neutral' });
await mermaid.run({ querySelector: '.mermaid' });
</script>`,
  'post'
);
