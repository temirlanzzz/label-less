/* Label Less — UI: controls, hand-drawn canvas chart, query grid, stat tiles. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const MAX_LABELS = 400;     // where a run stops; the interesting part of the curve
  const BASE_ITERS = 1500;    // all-labels baseline is trained to convergence
  const TICK_MS = 140;

  let data, sp, active, random, baseline = null, timer = null, running = false, peekCache = null;

  // ---------- theme ----------
  const themeBtn = $('theme');
  themeBtn.addEventListener('click', () => {
    const root = document.documentElement;
    const cur = root.getAttribute('data-theme') ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
    draw(); drawGrid();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => { draw(); drawGrid(); });

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  // ---------- setup ----------
  function settings() {
    const seed = Math.max(0, Math.min(9999, parseInt($('seed').value, 10) || 0));
    return { seed, strategy: $('strategy').value, batch: parseInt($('batch').value, 10) };
  }

  function reset() {
    stop();
    const s = settings();
    const init = AL.initialLabels(sp, s.seed);
    active = new AL.Learner(data, sp, init, { mode: 'active', strategy: s.strategy, batch: s.batch, seed: s.seed });
    random = new AL.Learner(data, sp, init, { mode: 'random', batch: s.batch, seed: s.seed });
    peekCache = null;
    render();
  }

  function stepOnce() {
    if (active.labeled.length >= MAX_LABELS || active.done()) { stop(); return false; }
    active.step(); random.step();
    peekCache = null;
    render();
    if (active.labeled.length >= MAX_LABELS) { stop(); return false; }
    return true;
  }

  function loop() {
    if (!running) return;
    const t0 = performance.now();
    if (!stepOnce()) return;
    timer = setTimeout(loop, Math.max(16, TICK_MS - (performance.now() - t0)));
  }
  function start() {
    if (active.labeled.length >= MAX_LABELS) reset();
    running = true; $('run').textContent = 'pause'; loop();
  }
  function stop() {
    running = false; clearTimeout(timer); $('run').textContent = 'run';
  }

  // ---------- render ----------
  const pct = (a) => (a == null ? '–' : (a * 100).toFixed(1) + '%');
  function firstReach(curve, t) { for (const [n, a] of curve) if (a >= t) return n; return null; }

  function render() {
    const a = active.curve[active.curve.length - 1], r = random.curve[random.curve.length - 1];
    $('t-labels').innerHTML = `${a[0]}<small> / ${MAX_LABELS}</small>`;
    $('t-active').textContent = pct(a[1]);
    $('t-random').textContent = pct(r[1]);
    const fa = firstReach(active.curve, 0.9), fr = firstReach(random.curve, 0.9);
    $('t-90').innerHTML = `${fa ?? '–'}<small> / </small>${fr ?? '–'}`;
    const s = settings();
    $('status').textContent = active.labeled.length >= MAX_LABELS
      ? `done · ${s.strategy} · seed ${s.seed}`
      : `${running ? 'running' : 'ready'} · ${s.strategy} · batch ${s.batch} · seed ${s.seed}`;
    draw(); drawGrid(); drawBars();
  }

  function drawBars() {
    const ca = new Array(10).fill(0), cr = new Array(10).fill(0);
    for (const i of active.labeled) ca[data.y[i]]++;
    for (const i of random.labeled) cr[data.y[i]]++;
    const m = Math.max(...ca, ...cr, 1);
    const el = $('bars');
    el.innerHTML = ca.map((v, k) => `<div class="bar" title="digit ${k}: active ${v}, random ${cr[k]}"><i class="a" style="height:${(v / m) * 100}%"></i><i class="r" style="height:${(cr[k] / m) * 100}%"></i></div>`).join('');
  }

  // ---------- chart (hand-drawn on canvas) ----------
  const chart = $('chart');
  function draw() {
    if (!active) return;
    const dpr = window.devicePixelRatio || 1;
    const W = chart.clientWidth, H = chart.clientHeight;
    if (chart.width !== Math.round(W * dpr) || chart.height !== Math.round(H * dpr)) {
      chart.width = Math.round(W * dpr); chart.height = Math.round(H * dpr);
    }
    const g = chart.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const ink3 = css('--ink-3'), line = css('--line'), accent = css('--accent'), rnd = css('--random'), ink = css('--ink');
    const mono = '11px ' + css('--mono');
    const pad = { l: 40, r: 12, t: 10, b: 28 };
    const lo = Math.min(...active.curve.map((c) => c[1]), ...random.curve.map((c) => c[1]));
    const x0 = 0, x1 = MAX_LABELS, y0 = Math.min(0.4, Math.floor(lo * 10) / 10), y1 = 1.0;
    const X = (v) => pad.l + (v - x0) / (x1 - x0) * (W - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (H - pad.t - pad.b);

    g.font = mono; g.lineWidth = 1;
    // grid
    g.fillStyle = ink3; g.textAlign = 'right'; g.textBaseline = 'middle';
    for (let v = y0; v <= 1.0001; v += 0.1) {
      g.strokeStyle = line; g.beginPath(); g.moveTo(pad.l, Math.round(Y(v)) + 0.5); g.lineTo(W - pad.r, Math.round(Y(v)) + 0.5); g.stroke();
      g.fillText(Math.round(v * 100) + '%', pad.l - 8, Y(v));
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    const step = W < 480 ? 100 : 50;
    for (let v = 0; v <= x1; v += step) g.fillText(String(v), X(v), H - pad.b + 8);

    // 90% target
    g.save(); g.strokeStyle = ink3; g.globalAlpha = 0.6; g.setLineDash([2, 3]);
    g.beginPath(); g.moveTo(pad.l, Math.round(Y(0.9)) + 0.5); g.lineTo(W - pad.r, Math.round(Y(0.9)) + 0.5); g.stroke(); g.restore();

    // all-labels baseline
    if (baseline != null) {
      g.save(); g.strokeStyle = ink3; g.lineWidth = 1.5; g.setLineDash([6, 5]);
      g.beginPath(); g.moveTo(pad.l, Y(baseline)); g.lineTo(W - pad.r, Y(baseline)); g.stroke(); g.restore();
      g.fillStyle = ink3; g.textAlign = 'left'; g.textBaseline = 'bottom';
      g.fillText('all labels ' + pct(baseline), pad.l + 26, Y(baseline) - 4);
    }

    const series = (curve, color, width) => {
      g.strokeStyle = color; g.lineWidth = width; g.lineJoin = 'round'; g.lineCap = 'round';
      g.beginPath();
      curve.forEach(([n, a], i) => { const px = X(n), py = Y(Math.max(y0, a)); i ? g.lineTo(px, py) : g.moveTo(px, py); });
      g.stroke();
      const [n, a] = curve[curve.length - 1];
      g.fillStyle = color; g.beginPath(); g.arc(X(n), Y(Math.max(y0, a)), 3.5, 0, Math.PI * 2); g.fill();
    };
    series(random.curve, rnd, 1.75);
    series(active.curve, accent, 2.25);

    // markers where each first crossed 90%
    const mark = (curve, color) => {
      const n = firstReach(curve, 0.9); if (n == null) return;
      g.strokeStyle = color; g.lineWidth = 1; g.globalAlpha = 0.5;
      g.beginPath(); g.moveTo(Math.round(X(n)) + 0.5, Y(0.9)); g.lineTo(Math.round(X(n)) + 0.5, H - pad.b); g.stroke();
      g.globalAlpha = 1; g.fillStyle = color; g.textAlign = 'left'; g.textBaseline = 'bottom';
      g.fillText(String(n), X(n) + 4, H - pad.b - 2);
    };
    mark(active.curve, accent); mark(random.curve, ink);
  }
  window.addEventListener('resize', () => draw());

  // ---------- query grid ----------
  const grid = $('grid');
  function drawGrid() {
    if (!active) return;
    const n = Math.min(active.batch, 20);
    if (!peekCache) peekCache = active.done() ? [] : active.peek(n);
    while (grid.children.length < n) {
      const c = document.createElement('div'); c.className = 'cell';
      c.innerHTML = '<canvas width="8" height="8"></canvas><div class="cap"></div>';
      grid.appendChild(c);
    }
    while (grid.children.length > n) grid.removeChild(grid.lastChild);
    const ink = hexToRgb(css('--ink')), bg = hexToRgb(css('--card'));
    peekCache.forEach((q, k) => {
      const cell = grid.children[k];
      const cv = cell.firstChild, g = cv.getContext('2d');
      const img = g.createImageData(8, 8);
      for (let p = 0; p < 64; p++) {
        const v = data.X[q.i * 65 + p];
        for (let c = 0; c < 3; c++) img.data[p * 4 + c] = Math.round(bg[c] + (ink[c] - bg[c]) * v);
        img.data[p * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      let guess = 0; for (let c = 1; c < 10; c++) if (q.p[c] > q.p[guess]) guess = c;
      const truth = data.y[q.i];
      cv.setAttribute('aria-label', `digit ${truth}, model guesses ${guess}`);
      cell.lastChild.innerHTML = `${guess}→<b class="${guess === truth ? '' : 'x'}">${truth}</b>`;
      cell.style.visibility = 'visible';
    });
    for (let k = peekCache.length; k < n; k++) grid.children[k].style.visibility = 'hidden';
  }
  function hexToRgb(h) {
    h = h.replace('#', '');
    if (h.length === 3) h = h.split('').map((c) => c + c).join('');
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }

  // ---------- baseline, computed in chunks so the page stays responsive ----------
  function computeBaseline() {
    const tr = AL.trainer(data, sp.pool);
    const chunk = () => {
      tr.run(100);
      if (tr.iters < BASE_ITERS) { setTimeout(chunk, 0); return; }
      baseline = AL.accuracy(tr.W, data, sp.test);
      draw();
    };
    setTimeout(chunk, 50);
  }

  // ---------- wire up ----------
  $('run').addEventListener('click', () => (running ? stop() : start()));
  $('step').addEventListener('click', () => { stop(); stepOnce(); });
  $('reset').addEventListener('click', reset);
  ['strategy', 'batch', 'seed'].forEach((id) => $(id).addEventListener('change', reset));

  fetch('digits.json')
    .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then((json) => {
      data = AL.decode(json);
      sp = AL.split(data);
      ['run', 'step', 'reset'].forEach((id) => ($(id).disabled = false));
      reset();
      computeBaseline();
      if (/[?&]autorun\b/.test(location.search)) start();
    })
    .catch((e) => { $('status').textContent = 'could not load digits.json — ' + e.message; });

  // bench() for the console / headless runs: AL.bench with the loaded data.
  window.bench = (opts) => {
    const t = performance.now();
    const res = AL.bench(data, opts);
    res.baseline300 = AL.allLabelsBaseline(data, sp, 300);
    res.baseline1500 = AL.allLabelsBaseline(data, sp, 1500);
    res.ms = Math.round(performance.now() - t);
    return res;
  };
})();
