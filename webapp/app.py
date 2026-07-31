"""FastAPI web demo for the Factor-Risk-Decomposition project.

Showcases the quant pipeline (factor analysis, backtest, PCA risk decomposition,
and a live synthetic-market stress test) as a single-page dashboard.

The app consumes the precomputed CSVs that the notebooks write to data/processed/
(and data/raw/ff_factors.csv), recomputes the light ML pieces once at startup
(PCA, Marchenko-Pastur cutoff, the Fama-French alpha, and the NB06 factor model
used by the live "generate a synthetic market" button), and serves JSON + static.

Local run:

    uvicorn webapp.app:app --host 127.0.0.1 --port 8055
"""
from __future__ import annotations

import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sklearn.decomposition import PCA

from frd.research import (
    FF_COLUMNS,
    ArtifactError,
    ArtifactSpec,
    block_indices,
    decile_long_returns,
    fama_french_alpha,
    fama_french_regression,
    marchenko_pastur,
    momentum_signal,
    series_metrics,
    validate_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
RAW = REPO_ROOT / "data" / "raw"
STATIC_DIR = REPO_ROOT / "webapp" / "static"
REQUIRED_ARTIFACTS = {
    "returns": ArtifactSpec(PROCESSED / "returns_monthly.csv"),
    "signal": ArtifactSpec(PROCESSED / "momentum_signal.csv"),
    "backtest": ArtifactSpec(PROCESSED / "backtest_returns.csv", ("long_net", "short_net", "ls_net", "long_excess", "ls_excess")),
    "ff": ArtifactSpec(RAW / "ff_factors.csv", tuple(FF_COLUMNS)),
    "sectors": ArtifactSpec(PROCESSED / "sector_mapping.csv", ("ticker", "sector")),
    "ic": ArtifactSpec(PROCESSED / "ic_monthly.csv"),
    "factor_corr": ArtifactSpec(PROCESSED / "factor_correlation.csv"),
    "variance": ArtifactSpec(PROCESSED / "variance_decomposition.csv", ("component", "variance", "pct")),
    "synthetic": ArtifactSpec(PROCESSED / "synthetic_backtest_results.csv", ("bootstrap_alpha",)),
}

# Headline numbers are recomputed from data at startup, not hardcoded, so the demo
# always matches whatever the notebooks last produced.


def _file_version(path: Path) -> str:
    try:
        st = path.stat()
        return f"{int(st.st_mtime)}-{st.st_size}"
    except FileNotFoundError:
        return "missing"


# --------------------------------------------------------------------------- #
# Startup: load data, recompute the ML pieces once for the process lifetime.
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    s = app.state
    try:
        validate_artifacts(REPO_ROOT, REQUIRED_ARTIFACTS)
    except ArtifactError as exc:
        raise RuntimeError(str(exc)) from exc

    # --- core panels ---
    s.returns = pd.read_csv(PROCESSED / "returns_monthly.csv", index_col=0, parse_dates=True)
    s.signal = pd.read_csv(PROCESSED / "momentum_signal.csv", index_col=0, parse_dates=True)
    s.backtest = pd.read_csv(PROCESSED / "backtest_returns.csv", index_col=0, parse_dates=True)
    s.ff = pd.read_csv(RAW / "ff_factors.csv", index_col=0, parse_dates=True)
    s.sectors = pd.read_csv(PROCESSED / "sector_mapping.csv")
    s.ic_monthly = pd.read_csv(PROCESSED / "ic_monthly.csv", index_col=0, parse_dates=True)
    s.factor_corr = pd.read_csv(PROCESSED / "factor_correlation.csv", index_col=0)
    s.variance_decomp = pd.read_csv(PROCESSED / "variance_decomposition.csv")
    s.synthetic = pd.read_csv(PROCESSED / "synthetic_backtest_results.csv")

    # --- overview: equity curves, drawdowns, FF alpha, headline metrics ---
    bt = s.backtest
    bt_plot = bt.dropna(subset=["long_net", "ls_net"])
    s.equity = {
        "dates": bt_plot.index.strftime("%Y-%m-%d").tolist(),
        "long": ((1 + bt_plot["long_net"]).cumprod() - 1).round(4).tolist(),
        "ls": ((1 + bt_plot["ls_net"]).cumprod() - 1).round(4).tolist(),
    }
    ew_monthly = s.returns.mean(axis=1).reindex(bt.index)
    ew_plot = ew_monthly.reindex(bt_plot.index)
    s.equity["ew"] = ((1 + ew_plot).cumprod() - 1).round(4).fillna(0).tolist()
    long_wealth = (1 + bt_plot["long_net"]).cumprod()
    long_dd = ((long_wealth - long_wealth.cummax()) / long_wealth.cummax()).round(4)
    s.drawdown = {"dates": bt_plot.index.strftime("%Y-%m-%d").tolist(), "long": long_dd.tolist()}

    # FF 4-factor alpha (re-fitted from long_excess, period-aligned)
    ff_pm = s.ff[FF_COLUMNS].copy()
    ff_pm.index = ff_pm.index.to_period("M")
    m = fama_french_regression(bt["long_net"], s.ff)
    s.headline = {
        "alpha_annual": float(m.params[0] * 12),
        "alpha_t": float(m.tvalues[0]),
        "alpha_p": float(m.pvalues[0]),
        "mkbeta": float(m.params[1]),
        "smb_beta": float(m.params[2]),
        "hml_beta": float(m.params[3]),
        "mom_beta": float(m.params[4]),
        "r_squared": float(m.rsquared),
        **series_metrics(bt["long_net"]),
    }
    s.headline_ew = series_metrics(ew_monthly)
    s.headline_ls = series_metrics(bt["ls_net"])
    active = (bt["long_net"] - ew_monthly).dropna()
    s.headline["active_annual"] = float(active.mean() * 12)
    s.headline["active_ir"] = float(active.mean() / active.std() * np.sqrt(12)) if active.std() > 0 else float("nan")

    # --- factors: IC summary, walk-forward, correlation ---
    ic = s.ic_monthly
    s.factor_ic = {
        f: {"mean": float(ic[f].mean()), "std": float(ic[f].std()),
            "ir": float(ic[f].mean() / ic[f].std() * np.sqrt(12)) if ic[f].std() > 0 else float("nan")}
        for f in ic.columns
    }
    # walk-forward: mean IC per 5y window (momentum + others), if enough dates
    wf = {}
    for start, end in [(2006, 2011), (2011, 2016), (2016, 2021), (2021, 2026)]:
        mask = (ic.index.year >= start) & (ic.index.year < end)
        sub = ic[mask]
        if len(sub) >= 12:
            wf[f"{start}-{end}"] = {f: float(sub[f].mean()) for f in ic.columns}
    s.walkforward = wf

    # --- PCA / risk: recompute on the balanced panel (mirrors NB05) ---
    panel = s.returns.dropna(thresh=240, axis=1).dropna()
    s.panel_tickers = panel.columns.tolist()
    Xc = panel.values
    T, N = Xc.shape
    mean_r = Xc.mean(axis=0)
    Xc = Xc - mean_r
    s.T, s.N = T, N
    n_comp = min(T, N)
    pca = PCA(n_components=n_comp).fit(Xc)
    eigvals = pca.explained_variance_                 # length n_comp
    s.eigvals = eigvals
    s.explained_pct = (pca.explained_variance_ratio_ * 100)
    B_all = pca.components_.T                          # (N, n_comp)
    s.mp = marchenko_pastur(eigvals, T, N)
    s.k = max(int(s.mp["signal_count"]), 1)
    s.B = B_all[:, : s.k]
    s.F = Xc @ s.B
    s.E = Xc - s.F @ s.B.T
    s.mean_r = mean_r
    # variance decomp (from the saved canonical table)
    vd = {row["component"]: row["pct"] for _, row in s.variance_decomp.iterrows()}
    s.var_decomp = {
        "systematic": float(vd.get("systematic", float("nan"))),
        "idiosyncratic": float(vd.get("idiosyncratic", float("nan"))),
    }
    # PC1/PC2 loadings + latest momentum score per ticker (for the explorer scatter)
    last_scores = s.signal.iloc[-2].reindex(s.panel_tickers)  # -2: trade t+1
    sec_map = dict(zip(s.sectors["ticker"], s.sectors["sector"]))
    s.ticker_rows = [
        {
            "ticker": t,
            "sector": sec_map.get(t, "Unknown"),
            "momentum": (None if pd.isna(last_scores[t]) else float(last_scores[t])),
            "pc1": float(B_all[i, 0]),
            "pc2": float(B_all[i, 1]),
        }
        for i, t in enumerate(s.panel_tickers)
    ]

    # --- stress test: real baseline + precomputed synthetic distribution ---
    ff_panel = ff_pm.reindex(panel.index.to_period("M"))
    if ff_panel[FF_COLUMNS].isna().any().any():
        raise RuntimeError("Fama-French factors do not cover the full balanced PCA panel.")
    ff_panel = ff_panel.reset_index(drop=True)
    s.ff_panel = ff_panel
    ret_real = pd.DataFrame(panel.values, columns=s.panel_tickers)
    real_alpha, _, _ = fama_french_alpha(decile_long_returns(momentum_signal(ret_real), ret_real), ff_panel)
    s.real_alpha = real_alpha
    s.synth_alphas = s.synthetic["bootstrap_alpha"].dropna().to_numpy()
    s.synth_mean = float(s.synth_alphas.mean())
    s.synth_p = float((s.synth_alphas >= real_alpha).mean())

    print(f"[webapp] startup ok: T={T} N={N} k={s.k} signal={s.mp['signal_count']} "
          f"alpha={s.headline['alpha_annual']*100:.2f}% p={s.synth_p:.2f}")
    yield


app = FastAPI(title="Factor Risk Decomposition", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _json_safe(o: Any) -> Any:
    """Recursively coerce numpy types / NaN for FastAPI JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if not np.isnan(v) else None
    if isinstance(o, np.ndarray):
        return [_json_safe(v) for v in o.tolist()]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css_v = _file_version(STATIC_DIR / "app.css")
    js_v = _file_version(STATIC_DIR / "app.js")
    html = re.sub(r"app\.css\?v=[^\s\"']+", f"app.css?v={css_v}", html)
    html = re.sub(r"app\.js\?v=[^\s\"']+", f"app.js?v={js_v}", html)
    return HTMLResponse(content=html)


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    # Disallow crawling — this is an interactive demo, not a site to index.
    return "User-agent: *\nDisallow: /\n"


@app.get("/api/health")
def health():
    s = app.state
    return {"ok": True, "T": s.T, "N": s.N, "signal_factors": s.mp["signal_count"],
            "alpha_annual": s.headline["alpha_annual"]}


@app.get("/api/overview")
def overview():
    s = app.state
    return _json_safe({
        "headline": s.headline,
        "ew": s.headline_ew,
        "long_short": s.headline_ls,
        "equity": s.equity,
        "drawdown": s.drawdown,
    })


@app.get("/api/factors")
def factors():
    s = app.state
    return _json_safe({
        "ic": s.factor_ic,
        "walkforward": s.walkforward,
        "correlation": {"labels": list(s.factor_corr.columns),
                        "matrix": s.factor_corr.values.tolist()},
    })


@app.get("/api/risk")
def risk():
    s = app.state
    top = 50
    return _json_safe({
        "spectrum": {"explained_pct": s.explained_pct[:top].tolist(),
                     "eigenvalue": s.eigvals[:top].tolist()},
        "mp": s.mp,
        "variance_decomp": s.var_decomp,
        "pc1_market_corr": None,  # placeholder (kept for forward-compat)
    })


@app.get("/api/tickers")
def tickers():
    return _json_safe(app.state.ticker_rows)


@app.get("/api/ticker/{ticker}")
def ticker_detail(ticker: str):
    ticker = ticker.upper()
    for row in app.state.ticker_rows:
        if row["ticker"] == ticker:
            return _json_safe(row)
    raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")


@app.get("/api/stress")
def stress_data():
    """Precomputed synthetic-market alpha distribution + the real baseline."""
    s = app.state
    return _json_safe({
        "real_alpha": s.real_alpha,
        "mean": s.synth_mean,
        "p_value": s.synth_p,
        "distribution": s.synth_alphas.tolist(),
    })


@app.post("/api/stress")
def stress_generate():
    """Generate ONE fresh synthetic market (block bootstrap) and re-run the backtest."""
    s = app.state
    rng = np.random.RandomState(int(time.time() * 1e6) % (2**31))
    idx = block_indices(s.T, L=15, rng=rng)
    R_synth = s.F[idx] @ s.B.T + s.E[idx] + s.mean_r
    ret_df = pd.DataFrame(R_synth, columns=s.panel_tickers)
    long_ret = decile_long_returns(momentum_signal(ret_df), ret_df)
    ff_synth = s.ff_panel.iloc[idx].reset_index(drop=True)
    alpha, tstat, _ = fama_french_alpha(long_ret, ff_synth)
    wealth = (1 + long_ret).cumprod()
    equity = (wealth - 1).round(4).tolist()
    percentile = float((s.synth_alphas <= alpha).mean()) if not np.isnan(alpha) else None
    return _json_safe({
        "alpha": alpha,
        "t": tstat,
        "equity": equity,
        "percentile": percentile,
        "real_alpha": s.real_alpha,
    })
