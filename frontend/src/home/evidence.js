// Evidence charts (content script section 6), drawn as SVG from src/evidence/evidence.json (sources listed in that file).
// Charts rules from the script: no backgrounds, no 3D, direct labels where they fit, axis labels 11px mono --text-3,
// gridlines --hairline, Depth Wizard in --accent, baselines in --text-3, values written on the bars.
// Every mark has a hover tooltip (<title>); every chart has a hidden data table for screen readers. The marks start
// collapsed (CSS) and grow once when the chart gets .is-drawn (main.js).
import data from './evidence/evidence.json';

const NS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs, parent, text) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  parent?.append(n);
  return n;
};
const tip = (mark, text) => { svgEl('title', {}, mark, text); return mark; };
const fmt = (v, d) => v.toFixed(d).replace('-', '−');
const name = (id) => id.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());   // tile id -> "Goa estuary"
// a bar with its top corners rounded (4px), anchored square on the baseline
const barPath = (x, y, w, h, r = 4) => {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
};
const legend = (host, items) => {
  const ul = document.createElement('ul'); ul.className = 'legend'; ul.setAttribute('aria-hidden', 'true');
  ul.innerHTML = items.map(([cls, label]) => `<li><i style="background: var(${cls})"></i>${label}</li>`).join('');
  host.append(ul);
};
// a long axis title as HTML under the plot, so it wraps on narrow screens
const axisTitle = (host, text) => { const p = document.createElement('p'); p.className = 'axis-title'; p.setAttribute('aria-hidden', 'true'); p.textContent = text; host.append(p); };
const table = (host, caption, head, rows) => {
  const wrap = document.createElement('div'), t = document.createElement('table'); wrap.className = 'sr-only'; wrap.append(t);
  t.innerHTML = `<caption>${caption}</caption><thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join('')}</tr></thead>`
    + `<tbody>${rows.map((r) => `<tr>${r.map((c, i) => (i ? `<td>${c}</td>` : `<th scope="row">${c}</th>`)).join('')}</tr>`).join('')}</tbody>`;
  host.append(wrap);   // the wrapper clips it: a table ignores the 1px box itself
};

// Interaction (owner 2026-10-16): hovering a metric in the bars or the waffles highlights it in both (the rest dims) and
// shows a tooltip; a click or tap pins it until the next click elsewhere. Linked across charts through data-metric.
let pinned = null;
const tipEl = Object.assign(document.createElement('div'), { className: 'viz-tip' });
tipEl.setAttribute('aria-hidden', 'true');
document.body.append(tipEl);
const focus = (metric) => {
  for (const el of document.querySelectorAll('.chart .metric')) el.classList.toggle('is-dim', !!metric && el.dataset.metric !== metric);
};
const showTip = (html, e) => {
  tipEl.innerHTML = html; tipEl.classList.add('is-on');
  const r = tipEl.getBoundingClientRect(), x = Math.min(e.clientX + 16, innerWidth - r.width - 8), y = e.clientY - r.height - 14;
  tipEl.style.translate = `${Math.max(8, x)}px ${y < 8 ? e.clientY + 20 : y}px`;
};
const hit = (target, metric, html) => {
  target.classList.add('hit');
  target.addEventListener('pointermove', (e) => { focus(metric); showTip(html, e); });
  target.addEventListener('pointerleave', () => { focus(pinned); if (!pinned) tipEl.classList.remove('is-on'); });
  target.addEventListener('click', (e) => { e.stopPropagation(); pinned = pinned === metric ? null : metric; focus(pinned || metric); showTip(html, e); if (!pinned) tipEl.classList.remove('is-on'); });
};
addEventListener('click', () => { if (pinned) { pinned = null; focus(null); tipEl.classList.remove('is-on'); } });
addEventListener('scroll', () => tipEl.classList.remove('is-on'), { passive: true });   // a pinned highlight stays, its tooltip goes

// Block A as a 2 x 2 table (owner 2026-10-16): each row pairs a bar panel with the waffles of its two metrics, error
// (MAE, RMSE) on top, agreement (Pearson, Spearman) below; columns "Depth Wizard vs oracle" and "Tiles won, of 50".
// Cells are HTML (landmark-row outlines); each draws its own SVG at its width. Hover/click is linked through data-metric.
const PANELS = [
  { title: 'Error, lower is better', names: ['MAE', 'RMSE'], max: 5, ticks: [0, 1, 2, 3, 4, 5], d: [3, 2], unit: ' m' },
  { title: 'Agreement, higher is better', names: ['Pearson', 'Spearman'], max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], d: [3, 3], unit: '' },
];
function barPanel(cell, W, p) {
  const rows = p.names.map((n) => data.blockA.metrics.find((x) => x.name === n));
  const H = 150, top = 30, axisW = 44, bw = 30, gw = Math.min((W - axisW) / rows.length, 170), plotR = axisW + gw * rows.length;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${top + H + 32}`, 'aria-hidden': 'true' }, cell);
  const y = (v) => top + H - (v / p.max) * H;
  for (const t of p.ticks) {
    svgEl('line', { x1: axisW, x2: plotR, y1: y(t), y2: y(t), class: 'grid' }, svg);
    svgEl('text', { x: axisW - 8, y: y(t) + 4, 'text-anchor': 'end' }, svg, String(t));
  }
  rows.forEach((r, i) => {
    const cx = axisW + gw * i + gw / 2, mg = svgEl('g', { 'data-metric': r.name, class: 'metric' }, svg);
    [['dw', r.dw], ['base', r.oracle]].forEach(([cls, v], j) => {
      const x = cx + (j ? 1 : -1 - bw), h = Math.max(1, (v / p.max) * H);   // 2px surface gap between the pair
      svgEl('path', { d: barPath(x, y(v), bw, h), class: `${cls} grow grow--y` }, mg);
      svgEl('text', { x: x + bw / 2, y: y(v) - 8, 'text-anchor': 'middle', class: 'val grow grow--o' }, mg, fmt(v, p.d[j]));
    });
    svgEl('text', { x: cx, y: top + H + 22, 'text-anchor': 'middle' }, mg, r.name + (p.unit ? ' (m)' : ''));
    const better = r.better === 'lower' ? `${Math.round(r.oracle / r.dw)}× lower error` : `${fmt(r.dw - r.oracle, 3).replace(/^(?!−)/, '+')} higher`;
    hit(svgEl('rect', { x: cx - gw / 2 + 4, y: top - 20, width: gw - 8, height: H + 52, fill: 'transparent' }, mg), r.name,
      `<b>${r.name}</b><span><i class="sw dw"></i>Depth Wizard ${fmt(r.dw, p.d[0])}${p.unit}</span><span><i class="sw base"></i>Oracle ${fmt(r.oracle, p.d[1])}${p.unit}</span><em>${better}</em>`);
  });
}
function wafflePair(cell, W, names, fit) {
  const w = data.wins, gp = 4, cols = 10, rows = Math.ceil(w.of / cols);
  // 17px squares (owner: 24 too big, 12 too small); one-column table: they shrink to fit the cell. Two columns: the cell's
  // spare width opens the space between the two waffles (owner: space between them, not bigger charts), up to 120px.
  const s = fit ? Math.max(8, Math.min(17, Math.floor(((W - 36) / 2 - (cols - 1) * gp) / cols))) : 17;
  const bw = cols * s + (cols - 1) * gp, bh = rows * s + (rows - 1) * gp, gap = fit ? 36 : Math.max(36, Math.min(120, Math.floor(W - 2 * bw)));   // at most 120px
  const svg = svgEl('svg', { viewBox: `0 0 ${2 * bw + gap} ${bh + 30}`, 'aria-hidden': 'true', style: `width: ${2 * bw + gap}px; max-width: 100%` }, cell);   // its own width: it sizes the waffle column
  names.forEach((n, k) => {
    const g = svgEl('g', { transform: `translate(${k * (bw + gap)},0)`, 'data-metric': n, class: 'metric' }, svg);
    for (let i = 0; i < w.of; i++) {
      const won = i < w[n], x = (i % cols) * (s + gp), y = Math.floor(i / cols) * (s + gp);
      const sq = svgEl('rect', { x, y, width: s, height: s, rx: 2, class: won ? 'dw grow grow--o' : 'grow grow--o', ...(won ? {} : { fill: 'none', stroke: 'var(--hairline)' }) }, g);
      sq.style.transitionDelay = `${i * 12}ms`;
    }
    svgEl('text', { x: 0, y: bh + 22, class: 'val' }, g, `${n}  ${w[n]} / ${w.of}`);
    hit(svgEl('rect', { x: -6, y: -6, width: bw + 12, height: bh + 36, fill: 'transparent' }, g), n,
      `<b>${n}</b><span><i class="sw dw"></i>Depth Wizard better on ${w[n]} of ${w.of} tiles</span><span><i class="sw none"></i>Oracle better on ${w.of - w[n]}</span>`);
  });
}
function matrix(host) {
  legend(host, [['--accent', 'Depth Wizard'], ['--text-3', 'Oracle-calibrated depth']]);
  const w = data.wins, counts = Object.entries(w).filter(([k]) => k !== 'of').map(([, v]) => v);
  const grid = document.createElement('div'); grid.className = 'matrix';
  grid.innerHTML = `<p class="matrix__head">Depth Wizard vs oracle</p><p class="matrix__head">Tiles won, of ${w.of} (${Math.min(...counts)}–${Math.max(...counts)} depending on the metric)</p>`;
  host.append(grid);
  const cells = PANELS.map((p) => {
    const bars = document.createElement('div'), waf = document.createElement('div');
    bars.className = 'matrix__cell'; waf.className = 'matrix__cell matrix__cell--waffles';
    bars.innerHTML = `<p class="matrix__title">${p.title}</p>`; waf.innerHTML = `<div class="matrix__waffles"><p class="matrix__title matrix__title--sub">${p.names.join(' and ')}</p></div>`;
    grid.append(bars, waf);
    return [bars, waf, p];
  });
  const inner = (el) => el.clientWidth - parseFloat(getComputedStyle(el).paddingLeft) - parseFloat(getComputedStyle(el).paddingRight);
  const oneCol = getComputedStyle(grid).gridTemplateColumns.split(' ').length === 1;
  for (const [bars, waf, p] of cells) { barPanel(bars, inner(bars), p); wafflePair(waf.firstChild, inner(waf), p.names, oneCol); }   // waffles under their label
  table(host, 'Depth Wizard against oracle-calibrated depth, DFC2019, 50 tiles', ['Metric', 'Depth Wizard', 'Oracle-calibrated depth', 'Tiles won, of 50'],
    data.blockA.metrics.map((r) => [r.name + (r.unit ? ` (${r.unit})` : ''), r.dw, r.oracle, w[r.name]]));
}

// Block B: one row per benchmark tile, GLO-30 and FABDEM error, sorted by FABDEM error
function dumbbell(host, W) {
  legend(host, [['--accent', 'FABDEM'], ['--text-3', 'Copernicus GLO-30']]);
  const rows = data.perTileRmse, labelW = W < 520 ? 96 : 128, top = 8;
  // in a pop-up the whole chart fits the window (owner: no scrolling): rows share what is left of the dialog's height
  const dlg = host.closest('dialog'), room = dlg ? dlg.clientHeight - (host.offsetTop - dlg.scrollTop) - 110 : Infinity;
  const rh = Math.max(10, Math.min(17, Math.floor(room / rows.length))), H = rows.length * rh;
  const max = Math.ceil(Math.max(...rows.map((r) => Math.max(r.fabdem, r.glo30)))), x = (v) => labelW + (v / max) * (W - labelW - 12);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${top + H + 24}`, 'aria-hidden': 'true' }, host);
  axisTitle(host, 'RMSE against ICESat-2 ground photons (m)');
  const step = max > 8 ? 2 : 1;
  for (let t = 0; t <= max; t += step) {
    svgEl('line', { x1: x(t), x2: x(t), y1: top, y2: top + H, class: 'grid' }, svg);
    svgEl('text', { x: x(t), y: top + H + 16, 'text-anchor': 'middle' }, svg, String(t));
  }
  rows.forEach((r, i) => {
    const cy = top + i * rh + rh / 2;
    svgEl('text', { x: labelW - 12, y: cy + 4, 'text-anchor': 'end', 'font-size': rh < 13 ? 10 : 11 }, svg, name(r.tile));
    const g = tip(svgEl('g', { class: 'grow grow--o' }, svg), `${name(r.tile)} (${r.category}): FABDEM ${fmt(r.fabdem, 2)} m, GLO-30 ${fmt(r.glo30, 2)} m`);
    g.style.transitionDelay = `${i * 18}ms`;
    svgEl('line', { x1: x(r.fabdem), x2: x(r.glo30), y1: cy, y2: cy, stroke: 'var(--text-3)', 'stroke-opacity': '.45', 'stroke-width': 2 }, g);
    svgEl('circle', { cx: x(r.glo30), cy, r: 4, class: 'base', stroke: 'var(--bg)', 'stroke-width': 2 }, g);
    svgEl('circle', { cx: x(r.fabdem), cy, r: 4, class: 'dw', stroke: 'var(--bg)', 'stroke-width': 2 }, g);
    svgEl('rect', { x: 0, y: cy - rh / 2, width: W, height: rh, fill: 'transparent' }, g);   // a hit target the full row high
  });
  table(host, 'Terrain error per benchmark tile in India, RMSE against ICESat-2 ground photons (m)', ['Tile', 'FABDEM', 'Copernicus GLO-30'],
    rows.map((r) => [name(r.tile), fmt(r.fabdem, 2), fmt(r.glo30, 2)]));
}

const charts = { matrix, dumbbell };
export function drawChart(host) {
  const W = Math.round(host.clientWidth), key = host.closest('dialog') ? `${W}x${innerHeight}` : String(W);
  if (!W || host.dataset.drawnW === key) return;
  host.dataset.drawnW = key;
  host.replaceChildren();
  charts[host.dataset.chart]?.(host, W);
}
