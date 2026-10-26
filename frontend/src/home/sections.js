// The content sections' behaviour (owner content script, 2026-10-16), shared by the page (src/main.js) and the standalone
// section export: letter splitting and the section titles' rise-in, text fade-ups, stat count-ups, and the evidence charts
// with their pop-up. Reduced motion: everything shown as is (CSS and the checks below).
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Split an element's text into words (.w, kept whole on a line) and characters (.c, numbered --ci); screen readers get the
// text whole from a .sr-only copy, the pieces are aria-hidden.
export function splitChars(el) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT), nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) node.replaceWith(...node.textContent.split(/(\s+)/).map((part) => {
    if (!part.trim()) return part;
    const w = document.createElement('span'); w.className = 'w';
    w.append(...[...part].map((ch) => { const c = document.createElement('span'); c.className = 'c'; c.textContent = ch; return c; }));
    return w;
  }));
  for (const w of el.querySelectorAll('.w')) w.setAttribute('aria-hidden', 'true');
  const sr = document.createElement('span'); sr.className = 'sr-only'; sr.textContent = el.textContent.replace(/\s+/g, ' ').trim(); el.prepend(sr);
  const chars = [...el.querySelectorAll('.c')];
  chars.forEach((c, i) => c.style.setProperty('--ci', i));
  return chars;
}


// Section titles (.h2, owner 2026-10-16): the statement's rise-in, made more obvious and replayed every time a title comes
// back into view (owner): when 45% of it is on screen, each character rises from 120% of its height below while fading in,
// .9s, 14ms apart (CSS, .rise.is-in); once it has fully left the screen it resets, unseen. All screen sizes; reduced
// motion: shown as is. Rows of cells marked .rise-cells play the same way, each cell floating up in turn (owner 2026-10-25),
// and the workflow labels are split too, for their letter-by-letter hover rise (CSS).
function setupTitleRise() {
  document.querySelectorAll('.pipe .h3').forEach(splitChars);
  for (const t of document.querySelectorAll('.h2')) { splitChars(t); t.classList.add('rise'); }
  const titles = [...document.querySelectorAll('.rise, .rise-cells')];
  if (reduceMotion) { titles.forEach((t) => t.classList.add('is-in')); return; }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.intersectionRatio >= 0.45) e.target.classList.add('is-in');
      else if (!e.isIntersecting) e.target.classList.remove('is-in');   // off screen: ready to play again
    }
  }, { threshold: [0, 0.45] });
  titles.forEach((t) => io.observe(t));
}

// New-section text fades up 24px as it enters the screen (.fade-up, CSS); reduced motion: shown at once (CSS).
function setupFadeUp() {
  const fadeUp = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add('is-in'); fadeUp.unobserve(e.target); }
}, { rootMargin: '0px 0px -10% 0px' });
  document.querySelectorAll('.fade-up').forEach((el) => fadeUp.observe(el));
}

// Stat numbers roll up (900ms, ease-out) once when they enter the screen, and again each time the mouse newly enters their
// tile (owner 2026-10-16; not while a roll is running). Reduced motion: shown as written.
function setupCountUp() {
  const nums = [...document.querySelectorAll('.stat__num[data-count]')];
  if (reduceMotion || !nums.length) return;
  const show = (el, v) => { const d = (el.dataset.count.split('.')[1] || '').length; el.textContent = v.toFixed(d) + (el.dataset.unit || ''); };
  const roll = (el) => {
    if (el.dataset.rolling) return;
    el.dataset.rolling = '1';
    const to = +el.dataset.count, t0 = performance.now();
    const tick = (now) => { const k = Math.min(1, (now - t0) / 900); show(el, to * (1 - (1 - k) ** 3)); if (k < 1) requestAnimationFrame(tick); else delete el.dataset.rolling; };
    requestAnimationFrame(tick);
  };
  nums.forEach((el) => show(el, 0));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { io.unobserve(e.target); roll(e.target); }
  }, { threshold: 0.5 });
  nums.forEach((el) => {
    io.observe(el);
    el.closest('.stat')?.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') roll(el); });
  });
}

// Evidence charts (src/evidence.js, its own chunk): loaded and drawn when a chart comes near, grown once when it is on
// screen, redrawn (already grown) if its width changes.
function setupCharts() {
  const charts = [...document.querySelectorAll('.chart[data-chart]')].filter((c) => !c.closest('dialog'));
  let mod = null;
  const load = () => (mod ||= import('./evidence.js'));
  const resize = new ResizeObserver((entries) => load().then(({ drawChart }) => entries.forEach((e) => drawChart(e.target))));
  // pop-up charts: drawn (and grown in) each time their dialog opens; Esc and the native close return focus to the button
  for (const open of document.querySelectorAll('.chart-open')) {
    const dialog = document.getElementById(open.getAttribute('aria-controls')), chart = dialog?.querySelector('.chart');
    if (!chart) continue;
    let y = 0;   // the page goes back exactly here on close: closing could drop it to the top in Firefox (owner 2026-10-25)
    dialog.addEventListener('close', () => scrollTo({ top: y, behavior: 'instant' }));
    open.addEventListener('click', () => {
      y = scrollY; dialog.showModal(); chart.classList.remove('is-drawn');
      load().then(({ drawChart }) => { delete chart.dataset.drawnW; drawChart(chart); resize.observe(chart); requestAnimationFrame(() => requestAnimationFrame(() => chart.classList.add('is-drawn'))); });
    });
    dialog.querySelector('.chart-dialog__close').addEventListener('click', () => dialog.close());
    addEventListener('resize', () => { if (dialog.open) load().then(({ drawChart }) => drawChart(chart)); }, { passive: true });   // refit to the new height
    dialog.addEventListener('click', (e) => {   // a click on the backdrop (outside the box; the box's own padding also reports the dialog)
      const r = dialog.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close();
    });
  }
  if (!charts.length) return;
  const near = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { near.unobserve(e.target); load().then(({ drawChart }) => { drawChart(e.target); resize.observe(e.target); }); }
  }, { rootMargin: '100% 0px' });
  const seen = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { seen.unobserve(e.target); load().then(() => requestAnimationFrame(() => requestAnimationFrame(() => e.target.classList.add('is-drawn')))); }
  }, { threshold: 0.3 });
  charts.forEach((c) => { near.observe(c); seen.observe(c); });
}

// Demo video (section 04, the protected #demo-video): the YouTube video's ID, the one setting. Empty: the frame keeps its
// "Recording in progress" placeholder. Otherwise the frame shows the video's poster and a play button, and the player
// (youtube-nocookie.com) loads only on a click (a facade: nothing from YouTube's player loads before), then plays. Opened by
// its deep link (/#demo-video), the player loads at once and plays muted with captions (browsers allow muted autoplay).
const DEMO_VIDEO_ID = '';

function setupDemoVideo() {
  const frame = document.querySelector('.demo-video');
  if (!frame || !DEMO_VIDEO_ID) return;
  const id = encodeURIComponent(DEMO_VIDEO_ID);
  const play = (params) => {
    const player = document.createElement('iframe');
    player.className = 'demo-video__player';
    player.src = `https://www.youtube-nocookie.com/embed/${id}?${params}`;
    player.title = 'Depth Wizard demo video';
    player.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
    player.allowFullscreen = true;
    player.referrerPolicy = 'strict-origin-when-cross-origin';
    frame.replaceChildren(player);
  };
  if (location.hash === '#demo-video') { play('autoplay=1&mute=1&playsinline=1&cc_load_policy=1'); return; }
  const poster = document.createElement('button');
  poster.type = 'button'; poster.className = 'demo-video__poster'; poster.setAttribute('aria-label', 'Play the demo video');
  poster.style.backgroundImage = `url("https://i.ytimg.com/vi/${id}/hqdefault.jpg")`;
  poster.innerHTML = '<span class="demo-video__play" aria-hidden="true"></span>';
  poster.addEventListener('click', () => play('autoplay=1&playsinline=1'), { once: true });
  frame.replaceChildren(poster);
}

// Feature and use-case cells in one row (owner 2026-10-17): the line under a title shows on hover (CSS) or on a click, tap,
// Enter or Space, which opens that cell and closes the others in its row.
function setupFeatureCells() {
  for (const row of document.querySelectorAll('.feature-row')) {
    const cells = [...row.querySelectorAll('.card')];
    const toggle = (cell) => { const open = !cell.classList.contains('is-open'); cells.forEach((c) => { c.classList.toggle('is-open', c === cell && open); c.setAttribute('aria-expanded', String(c === cell && open)); }); };
    for (const cell of cells) {
      cell.tabIndex = 0; cell.setAttribute('role', 'button'); cell.setAttribute('aria-expanded', 'false');
      cell.addEventListener('click', () => toggle(cell));
      cell.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(cell); } });
    }
  }
}

// The closing title's gradient runs across the whole title, not letter by letter (owner 2026-10-17): each letter's background
// is the title's size, shifted by the letter's place in it (the title is their offsetParent; offsets ignore the rise).
function setupLineGradient() {
  const t = document.querySelector('.cta__title'); if (!t) return;
  const fit = () => {
    t.style.setProperty('--W', `${t.offsetWidth}px`); t.style.setProperty('--H', `${t.offsetHeight}px`);
    for (const c of t.querySelectorAll('.c')) { c.style.setProperty('--bx', `${c.offsetLeft}px`); c.style.setProperty('--by', `${c.offsetTop}px`); }
  };
  new ResizeObserver(fit).observe(t); document.fonts.ready.then(fit);
}

export function setupSections() { setupTitleRise(); setupFadeUp(); setupCountUp(); setupCharts(); setupFeatureCells(); setupLineGradient(); setupDemoVideo(); }
