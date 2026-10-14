/* Label Less — active learning core.
 * Softmax regression, uncertainty sampling and a benchmark, written from scratch.
 * No dependencies. Works in the browser (window.AL) and in Node (module.exports).
 */
(function (root) {
  'use strict';

  const D = 65;            // 64 pixels + bias
  const K = 10;            // classes
  const ALPHABET = '0123456789abcdefg';

  // ---------- data ----------
  function decode(json) {
    const n = json.X.length;
    const X = new Float32Array(n * D);
    const y = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const s = json.X[i];
      for (let j = 0; j < 64; j++) X[i * D + j] = ALPHABET.indexOf(s[j]) / 16;
      X[i * D + 64] = 1; // bias
      y[i] = json.y.charCodeAt(i) - 48;
    }
    return { X, y, n };
  }

  // Deterministic PRNG (mulberry32)
  function rng(seed) {
    let a = (seed >>> 0) + 0x9e3779b9;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffle(arr, rand) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  // Fixed split: 500 test images, the remaining 1,297 are the unlabeled pool.
  // The split never changes with the run seed, so every run is scored on the same test set.
  function split(data, testSize = 500) {
    const idx = shuffle(Array.from({ length: data.n }, (_, i) => i), rng(12345));
    return { test: idx.slice(0, testSize), pool: idx.slice(testSize) };
  }

  // ---------- softmax regression ----------
  // Full-batch gradient descent with Nesterov momentum and L2 on the weights (not the bias).
  // trainer() keeps its state so long fits can be run in chunks without blocking the page.
  function trainer(data, idx, opts = {}) {
    const lr = opts.lr ?? 0.5;
    const l2 = opts.l2 ?? 1e-3;
    const mu = 0.9;
    const { X, y } = data;
    const n = idx.length;
    const W = new Float32Array(K * D);
    const V = new Float32Array(K * D);
    const G = new Float32Array(K * D);
    const Wl = new Float32Array(K * D);
    const z = new Float64Array(K);
    let done = 0;
    function run(iters) {
      for (let it = 0; it < iters; it++) {
        for (let q = 0; q < K * D; q++) Wl[q] = W[q] + mu * V[q]; // look-ahead point
        G.fill(0);
        for (let s = 0; s < n; s++) {
          const o = idx[s] * D;
          let m = -Infinity;
          for (let k = 0; k < K; k++) {
            let a = 0; const w = k * D;
            for (let j = 0; j < D; j++) a += Wl[w + j] * X[o + j];
            z[k] = a; if (a > m) m = a;
          }
          let sum = 0;
          for (let k = 0; k < K; k++) { z[k] = Math.exp(z[k] - m); sum += z[k]; }
          const t = y[idx[s]];
          for (let k = 0; k < K; k++) {
            const g = z[k] / sum - (k === t ? 1 : 0);
            if (g === 0) continue;
            const w = k * D;
            for (let j = 0; j < D; j++) G[w + j] += g * X[o + j];
          }
        }
        for (let k = 0; k < K; k++) for (let j = 0; j < D; j++) {
          const q = k * D + j;
          let g = G[q] / n;
          if (j < 64) g += l2 * Wl[q];
          V[q] = mu * V[q] - lr * g;
          W[q] += V[q];
        }
      }
      done += iters;
      return api;
    }
    const api = { W, run, get iters() { return done; } };
    return api;
  }

  function train(data, idx, opts = {}) {
    return trainer(data, idx, opts).run(opts.iters ?? 300).W;
  }

  function probs(W, X, i, out) {
    const o = i * D;
    let m = -Infinity;
    for (let k = 0; k < K; k++) {
      let a = 0; const w = k * D;
      for (let j = 0; j < D; j++) a += W[w + j] * X[o + j];
      out[k] = a; if (a > m) m = a;
    }
    let sum = 0;
    for (let k = 0; k < K; k++) { out[k] = Math.exp(out[k] - m); sum += out[k]; }
    for (let k = 0; k < K; k++) out[k] /= sum;
    return out;
  }

  function accuracy(W, data, idx) {
    const p = new Float64Array(K);
    let ok = 0;
    for (const i of idx) {
      probs(W, data.X, i, p);
      let b = 0;
      for (let k = 1; k < K; k++) if (p[k] > p[b]) b = k;
      if (b === data.y[i]) ok++;
    }
    return ok / idx.length;
  }

  // ---------- uncertainty ----------
  // Higher score = less sure = ask for this label first.
  const STRATEGIES = {
    margin(p) {
      let a = -1, b = -1;
      for (let k = 0; k < K; k++) { if (p[k] > a) { b = a; a = p[k]; } else if (p[k] > b) b = p[k]; }
      return -(a - b);
    },
    entropy(p) {
      let h = 0;
      for (let k = 0; k < K; k++) if (p[k] > 1e-12) h -= p[k] * Math.log(p[k]);
      return h;
    },
    least(p) {
      let a = 0;
      for (let k = 0; k < K; k++) if (p[k] > a) a = p[k];
      return 1 - a;
    },
  };

  function query(W, data, unlabeled, batch, strategy) {
    const p = new Float64Array(K);
    const f = STRATEGIES[strategy];
    const scored = unlabeled.map((i) => {
      probs(W, data.X, i, p);
      return { i, s: f(p), p: Array.from(p) };
    });
    scored.sort((a, b) => b.s - a.s || a.i - b.i);
    return scored.slice(0, batch);
  }

  // ---------- one learner ----------
  // mode: 'active' or 'random'. Both start from the same `init` labels.
  class Learner {
    constructor(data, sp, init, opts) {
      this.data = data;
      this.test = sp.test;
      this.mode = opts.mode;
      this.strategy = opts.strategy || 'margin';
      this.batch = opts.batch || 10;
      this.iters = opts.iters || 300;
      this.labeled = init.slice();
      const inL = new Set(init);
      this.unlabeled = sp.pool.filter((i) => !inL.has(i));
      if (this.mode === 'random') shuffle(this.unlabeled, rng(opts.seed * 7919 + 1));
      this.curve = [];
      this.last = null;
      this.fit();
    }
    fit() {
      this.W = train(this.data, this.labeled, { iters: this.iters });
      const acc = accuracy(this.W, this.data, this.test);
      this.curve.push([this.labeled.length, acc]);
      return acc;
    }
    done() { return this.unlabeled.length === 0; }
    step() {
      if (this.done()) return null;
      const b = Math.min(this.batch, this.unlabeled.length);
      let picked;
      if (this.mode === 'active') {
        picked = query(this.W, this.data, this.unlabeled, b, this.strategy);
        const take = new Set(picked.map((q) => q.i));
        this.unlabeled = this.unlabeled.filter((i) => !take.has(i));
      } else {
        picked = this.unlabeled.splice(0, b).map((i) => ({ i }));
      }
      // The "oracle" is simulated: it reveals the ground-truth label.
      for (const q of picked) this.labeled.push(q.i);
      this.last = picked;
      return this.fit();
    }
    // What would the active learner ask for next (without committing)?
    peek(n) { return query(this.W, this.data, this.unlabeled, n, this.strategy); }
  }

  function initialLabels(sp, seed, n = 10) {
    return shuffle(sp.pool.slice(), rng(seed)).slice(0, n);
  }

  function allLabelsBaseline(data, sp, iters = 1500) {
    return accuracy(train(data, sp.pool, { iters }), data, sp.test);
  }

  // ---------- benchmark ----------
  function firstReach(curve, t) {
    for (const [n, a] of curve) if (a >= t) return n;
    return null;
  }
  function accAt(curve, n) {
    for (const [m, a] of curve) if (m === n) return a;
    return null;
  }

  /**
   * bench({ seeds, strategy, batch, maxLabels }) runs active vs random from the
   * same 10 starting labels for each seed and reports labels-to-target and accuracy.
   */
  function bench(data, opts = {}) {
    const seeds = opts.seeds ?? [0, 1, 2, 3, 4];
    const strategy = opts.strategy ?? 'margin';
    const batch = opts.batch ?? 10;
    const maxLabels = opts.maxLabels ?? 400;
    const sp = split(data);
    const runs = [];
    for (const seed of seeds) {
      const init = initialLabels(sp, seed);
      const a = new Learner(data, sp, init, { mode: 'active', strategy, batch, seed });
      const r = new Learner(data, sp, init, { mode: 'random', batch, seed });
      while (a.labeled.length < maxLabels) { a.step(); r.step(); }
      runs.push({
        seed,
        active: { to90: firstReach(a.curve, 0.9), to95: firstReach(a.curve, 0.95), at100: accAt(a.curve, 100), at200: accAt(a.curve, 200), final: a.curve[a.curve.length - 1][1] },
        random: { to90: firstReach(r.curve, 0.9), to95: firstReach(r.curve, 0.95), at100: accAt(r.curve, 100), at200: accAt(r.curve, 200), final: r.curve[r.curve.length - 1][1] },
      });
    }
    return { strategy, batch, maxLabels, pool: sp.pool.length, test: sp.test.length, runs };
  }

  const AL = { decode, split, rng, shuffle, trainer, train, probs, accuracy, query, Learner, initialLabels, allLabelsBaseline, bench, STRATEGIES };
  if (typeof module !== 'undefined' && module.exports) module.exports = AL;
  else root.AL = AL;
})(typeof window !== 'undefined' ? window : globalThis);
