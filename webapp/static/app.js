/*
 * Browser-side controller for the Factor-Risk-Decomposition demo.
 *
 * The server returns precomputed results + a live synthetic-market generator as
 * JSON. This file renders the dashboard with Plotly.js, themed to the same Tokyo
 * Night palette as the rest of the site (transparent paper, Inter font).
 */

// Tokyo Night palette (matches app.css tokens).
const COLORS = {
  long: "#9ECE6A", ew: "#7aa2f7", ls: "#F7768E", accent: "#BB9AF7",
  real: "#E0AF68", muted: "#787C99", neg: "#F7768E", pos: "#9ECE6A",
  bar: "#2AC3DE", grid: "rgba(120,124,153,0.18)", fg: "#A9B1D6",
};
const FONT = { family: "Inter, sans-serif", size: 12, color: COLORS.fg };
const baseLayout = (extra = {}) => Object.assign({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: FONT,
  margin: { l: 52, r: 18, t: 16, b: 38 },
  xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid, linecolor: COLORS.grid, tickfont: { size: 10 } },
  yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid, linecolor: COLORS.grid, tickfont: { size: 10 } },
  legend: { font: { color: COLORS.fg }, orientation: "h", y: -0.2 },
  colorway: [COLORS.bar, COLORS.long, COLORS.accent, COLORS.ls, COLORS.real],
}, extra);
const CONFIG = { responsive: true, displayModeBar: false };
const $ = (id) => document.getElementById(id);

async function fetchJson(url, opts) {
  const r = await fetch(url, opts);
  const t = await r.text();
  let payload = {};
  try { payload = t ? JSON.parse(t) : {}; }
  catch {
    if (!r.ok) throw new Error(t || `HTTP ${r.status}`);
    throw new Error(`Invalid JSON from ${url}`);
  }
  if (!r.ok) {
    const detail = payload.detail || payload.message || t;
    throw new Error(detail || `HTTP ${r.status}`);
  }
  return payload;
}

const pct = (x, n = 2) => (x == null || isNaN(x)) ? "—" : (x * 100).toFixed(n) + "%";
const fixed = (x, n = 2) => (x == null || isNaN(x)) ? "—" : Number(x).toFixed(n);
const cls = (x) => (x == null || isNaN(x)) ? "" : (x >= 0 ? "pos" : "neg");

function tile(label, value, sub, valueClass = "") {
  return `<div class="metric"><div class="label">${label}</div>` +
         `<div class="value ${valueClass}">${value}</div>` +
         (sub ? `<div class="sub">${sub}</div>` : "") + `</div>`;
}

let overviewCache = null;

// --------------------------------------------------------------------------- //
async function init() {
  try {
    const h = await fetchJson("/api/health");
    $("health").textContent = `T=${h.T} · N=${h.N} · ${h.signal_factors} signal factors · α=${pct(h.alpha_annual)}`;
  } catch (e) { $("health").textContent = "offline"; }

  renderHero();
  renderStress();
  renderOverview();
  renderFactors();
  renderRisk();
  renderExplorer();
}

// ------------------------------- hero ------------------------------------- //
async function renderHero() {
  const d = await fetchJson("/api/overview");
  overviewCache = d;
  const hl = d.headline;
  const alphaPct = pct(hl.alpha_annual);
  const tStat = fixed(hl.alpha_t);

  $("hero-main").innerHTML =
    `<div class="hero-alpha ${cls(hl.alpha_annual)}">${alphaPct}</div>` +
    `<div class="hero-label">Fama-French 4-factor alpha (annualized)</div>` +
    `<div class="hero-context">t = ${tStat}  ·  R² = ${fixed(hl.r_squared)}  ·  net of 5 bps costs</div>` +
    `<p class="hero-blurb">This is the average return left after controlling for market, size, value, and momentum benchmark factors. The stock-ranking IC is weak, so this alpha is treated as evidence to stress-test, not a victory lap.</p>`;

  $("hero-action").innerHTML = "";
}

// ----------------------------- overview ------------------------------------ //
async function renderOverview() {
  const d = overviewCache || await fetchJson("/api/overview");
  const hl = d.headline;
  $("overview-metrics").innerHTML =
    tile("Sharpe (long-only)", fixed(hl.sharpe), `ann. return ${pct(hl.ann_return)}`) +
    tile("Max drawdown", pct(hl.max_drawdown), `vol ${pct(hl.ann_vol)}`) +
    tile("Active vs EW", pct(hl.active_annual), `IR ${fixed(hl.active_ir)}`, cls(hl.active_annual)) +
    tile("EW universe Sharpe", fixed(d.ew.sharpe), `ann. return ${pct(d.ew.ann_return)}`);

  const dates = d.equity.dates;
  Plotly.newPlot($("chart-equity"), [
    { x: dates, y: d.equity.long, name: "Long-only", mode: "lines", line: { color: COLORS.long, width: 2 } },
    { x: dates, y: d.equity.ew, name: "EW universe", mode: "lines", line: { color: COLORS.ew, width: 1.5 } },
    { x: dates, y: d.equity.ls, name: "Long-short", mode: "lines", line: { color: COLORS.ls, width: 1.5, dash: "dot" } },
  ], baseLayout({ yaxis: { title: "cumulative return", gridcolor: COLORS.grid }, legend: {} }), CONFIG);

  Plotly.newPlot($("chart-drawdown"), [
    { x: d.drawdown.dates, y: d.drawdown.long, name: "drawdown", type: "scatter", fill: "tozeroy",
      line: { color: COLORS.ls, width: 1 }, fillcolor: "rgba(247,118,142,0.25)" },
  ], baseLayout({ yaxis: { title: "drawdown", gridcolor: COLORS.grid, tickformat: ".0%" } }), CONFIG);
}

// ------------------------------ factors ------------------------------------ //
async function renderFactors() {
  const d = await fetchJson("/api/factors");
  const names = Object.keys(d.ic);
  Plotly.newPlot($("chart-ic"), [{
    x: names, y: names.map((n) => d.ic[n].mean), type: "bar",
    marker: { color: names.map((n) => n === "momentum" ? COLORS.pos : COLORS.bar) },
    text: names.map((n) => `IR=${fixed(d.ic[n].ir)}`), textposition: "outside",
  }], baseLayout({
    yaxis: { title: "mean monthly IC", gridcolor: COLORS.grid, zerolinecolor: COLORS.muted },
    shapes: [{ type: "line", x0: -0.5, x1: names.length - 0.5, y0: 0, y1: 0, line: { color: COLORS.muted, width: 1 } }],
  }), CONFIG);

  Plotly.newPlot($("chart-corr"), [{
    z: d.correlation.matrix, x: d.correlation.labels, y: d.correlation.labels,
    type: "heatmap", colorscale: [[0, "#F7768E"], [0.5, "#1A1B26"], [1, "#9ECE6A"]],
    zmin: -1, zmax: 1, showscale: false,
    text: d.correlation.matrix.map((r) => r.map((v) => v.toFixed(2))),
    texttemplate: "%{text}",
  }], baseLayout({ margin: { l: 64, r: 18, t: 16, b: 40 } }), CONFIG);

  const periods = Object.keys(d.walkforward);
  Plotly.newPlot($("chart-walkforward"), names.map((n) => ({
    name: n, type: "bar",
    x: periods, y: periods.map((p) => d.walkforward[p][n]),
  })), baseLayout({
    barmode: "group",
    yaxis: { title: "mean IC", gridcolor: COLORS.grid, zerolinecolor: COLORS.muted },
    shapes: [{ type: "line", x0: -0.5, x1: periods.length - 0.5, y0: 0, y1: 0, line: { color: COLORS.muted, width: 1 } }],
  }), CONFIG);
}

// ------------------------------- risk -------------------------------------- //
async function renderRisk() {
  const d = await fetchJson("/api/risk");
  const mp = d.mp;
  $("risk-metrics").innerHTML =
    tile("Significant factors", mp.signal_count, `above MP λ<sub>+</sub> = ${fixed(mp.lam_plus)}`) +
    tile("q = T / N", fixed(mp.q), `σ² = ${fixed(mp.sigma2)}`) +
    tile("Systematic risk", pct(d.variance_decomp.systematic / 100), "variance in top-k subspace", "pos") +
    tile("Idiosyncratic risk", pct(d.variance_decomp.idiosyncratic / 100), "orthogonal complement", "neg");

  const xs = d.spectrum.eigenvalue.map((_, i) => i + 1);
  Plotly.newPlot($("chart-scree"), [
    { x: xs, y: d.spectrum.eigenvalue, type: "bar", name: "eigenvalue", marker: { color: COLORS.bar } },
  ], baseLayout({
    yaxis: { title: "eigenvalue", gridcolor: COLORS.grid },
    xaxis: { title: "principal component", gridcolor: COLORS.grid },
    shapes: [{ type: "line", x0: 0, x1: xs.length, y0: mp.lam_plus, y1: mp.lam_plus,
              line: { color: COLORS.real, width: 2, dash: "dash" } }],
    annotations: [{ x: xs.length, y: mp.lam_plus, xanchor: "right", yanchor: "bottom",
                    text: "λ₊ (MP cutoff)", font: { color: COLORS.real, size: 10 }, showarrow: false }],
  }), CONFIG);

  Plotly.newPlot($("chart-vardecomp"), [{
    labels: ["Systematic", "Idiosyncratic"],
    values: [d.variance_decomp.systematic, d.variance_decomp.idiosyncratic],
    type: "pie", hole: 0.55,
    marker: { colors: [COLORS.bar, COLORS.ls] },
    textinfo: "label+percent", textfont: { color: COLORS.fg, size: 11 },
  }], baseLayout({ margin: { l: 10, r: 10, t: 10, b: 10 } }), CONFIG);
}

// ------------------------------ stress ------------------------------------- //
let stressDist = null;
async function renderStress() {
  const d = await fetchJson("/api/stress");
  stressDist = d;
  $("stress-metrics").innerHTML =
    tile("Real alpha (baseline)", pct(d.real_alpha), "raw-momentum pipeline", cls(d.real_alpha)) +
    tile("Bootstrap mean", pct(d.mean), "across 300 paths") +
    tile("Share ≥ real", pct(d.p_value), "higher = more typical (not a lucky path)");
  drawStressDist(null);
  $("stress-btn").addEventListener("click", generateStress);
}

function drawStressDist(genAlpha) {
  const shapes = [{ type: "line", x0: stressDist.real_alpha, x1: stressDist.real_alpha,
                    y0: 0, y1: 1, yref: "paper", line: { color: COLORS.real, width: 2 } }];
  if (genAlpha != null && !isNaN(genAlpha))
    shapes.push({ type: "line", x0: genAlpha, x1: genAlpha, y0: 0, y1: 1, yref: "paper",
                  line: { color: COLORS.ls, width: 2, dash: "dot" } });
  Plotly.newPlot($("chart-stress-dist"), [{
    x: stressDist.distribution, type: "histogram", name: "synthetic α",
    marker: { color: COLORS.bar, opacity: 0.65 },
    xbins: { size: 0.01 },
  }], baseLayout({
    xaxis: { title: "annualized FF alpha", tickformat: ".0%", gridcolor: COLORS.grid },
    yaxis: { title: "# synthetic markets", gridcolor: COLORS.grid }, shapes,
    annotations: [
      { x: stressDist.real_alpha, y: 1, yref: "paper", xanchor: "left", yanchor: "top",
        text: "real", font: { color: COLORS.real, size: 10 }, showarrow: false },
      ...(genAlpha != null && !isNaN(genAlpha) ? [{ x: genAlpha, y: 0.9, yref: "paper", xanchor: "left",
        text: "generated", font: { color: COLORS.ls, size: 10 }, showarrow: false }] : []),
    ],
  }), CONFIG);
}

async function generateStress() {
  const btn = $("stress-btn"); const ro = $("stress-readout");
  btn.disabled = true; ro.textContent = "generating…";
  try {
    const r = await fetchJson("/api/stress", { method: "POST" });
    drawStressDist(r.alpha);
    const xs = r.equity.map((_, i) => i);
    Plotly.newPlot($("chart-stress-equity"), [{
      x: xs, y: r.equity, type: "scatter", mode: "lines", fill: "tozeroy",
      line: { color: COLORS.ls, width: 1.5 }, fillcolor: "rgba(247,118,142,0.18)",
    }], baseLayout({
      xaxis: { title: "synthetic month", gridcolor: COLORS.grid },
      yaxis: { title: "cumulative return", gridcolor: COLORS.grid },
      annotations: [{ x: 0.5, y: 1.08, yref: "paper", xref: "paper", text:
        `α = ${pct(r.alpha)} · t = ${fixed(r.t)} · ${fixed((r.percentile || 0) * 100, 0)}th pctile`,
        font: { color: COLORS.real, size: 11 }, showarrow: false }],
    }), CONFIG);
    ro.textContent = `α = ${pct(r.alpha)}  (${fixed((r.percentile || 0) * 100, 0)}th percentile)`;
    const heroRo = $("hero-stress-readout");
    if (heroRo) heroRo.textContent = `Last generated: α = ${pct(r.alpha)}  (${fixed((r.percentile || 0) * 100, 0)}th percentile)`;
  } catch (e) { ro.textContent = "error: " + e.message; }
  btn.disabled = false;
}

// ----------------------------- explorer ------------------------------------ //
async function renderExplorer() {
  const rows = await fetchJson("/api/tickers");
  const sectors = [...new Set(rows.map((r) => r.sector))].sort();
  const palette = [COLORS.bar, COLORS.long, COLORS.accent, COLORS.ls, COLORS.real,
                   "#7aa2f7", "#B4F9F8", "#e0af68", "#f7768e", "#bb9af7", "#9ece6a"];
  const bySector = {};
  rows.forEach((r) => (bySector[r.sector] ||= []).push(r));
  const traces = sectors.map((sec, i) => ({
    name: sec, type: "scatter", mode: "markers",
    x: bySector[sec].map((r) => r.pc1), y: bySector[sec].map((r) => r.pc2),
    text: bySector[sec].map((r) => r.ticker),
    marker: { color: palette[i % palette.length], size: bySector[sec].map((r) => 5 + 3 * Math.min(Math.abs(r.momentum || 0), 3)),
              opacity: 0.75, line: { width: 0 } },
  }));
  Plotly.newPlot($("chart-scatter"), traces, baseLayout({
    xaxis: { title: "PC1 (≈ market)", gridcolor: COLORS.grid },
    yaxis: { title: "PC2 (≈ value)", gridcolor: COLORS.grid },
    legend: { font: { color: COLORS.fg, size: 9 }, orientation: "v", x: 1.02 },
    margin: { l: 52, r: 120, t: 16, b: 38 },
  }), CONFIG);

  const sel = $("ticker-select");
  rows.map((r) => r.ticker).sort().forEach((t) => {
    const o = document.createElement("option"); o.value = t; o.textContent = t; sel.appendChild(o);
  });
  sel.addEventListener("change", () => showTicker(rows.find((r) => r.ticker === sel.value), sel.value));
  // click a point to select that ticker
  $("chart-scatter").on("plotly_click", (ev) => {
    const p = ev.points[0]; const t = p.text;
    sel.value = t; showTicker(rows.find((r) => r.ticker === t), t);
  });
  sel.value = "AAPL"; showTicker(rows.find((r) => r.ticker === "AAPL"), "AAPL");
}

function showTicker(r, t) {
  const detail = $("ticker-detail");
  detail.replaceChildren();
  if (!r) {
    detail.textContent = `${t}: not in the balanced panel.`;
    return;
  }
  const ticker = document.createElement("b");
  ticker.textContent = r.ticker;
  const momentum = document.createElement("b");
  momentum.textContent = fixed(r.momentum);
  detail.append(
    ticker,
    document.createTextNode(` · sector: ${r.sector} · momentum z-score: `),
    momentum,
    document.createTextNode(` · PC1 (market): ${fixed(r.pc1)} · PC2 (value): ${fixed(r.pc2)}`),
  );
}

init();
