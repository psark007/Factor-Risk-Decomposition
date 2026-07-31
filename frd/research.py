"""Reusable finance and validation helpers for the project.

The notebooks remain the narrative surface, but core arithmetic lives here so
the research pipeline, dashboard, and tests do not drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm

FF_FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "Mom"]
FF_COLUMNS = FF_FACTOR_COLUMNS + ["RF"]


class ArtifactError(RuntimeError):
    """Raised when notebook-generated CSV artifacts are missing or malformed."""


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    columns: tuple[str, ...] = ()


def momentum_signal(ret_df: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum: trailing 11 monthly returns, shifted one month."""
    return ret_df.rolling(11).sum().shift(1)


def block_indices(T: int, L: int, rng: np.random.RandomState) -> np.ndarray:
    """Stationary block bootstrap: T time indices in variable-length blocks."""
    if T <= 0:
        raise ValueError("T must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    idx: list[int] = []
    while len(idx) < T:
        start = rng.randint(T)
        blen = rng.geometric(1.0 / L)
        idx.extend(((start + np.arange(blen)) % T).tolist())
    return np.array(idx[:T], dtype=int)


def equal_weights(tickers: pd.Index | list[str]) -> pd.Series:
    """Equal-weight vector for a selected set of tickers."""
    tickers = pd.Index(tickers)
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)


def turnover_from_weights(prev: pd.Series | None, curr: pd.Series) -> float:
    """One-way turnover from prior weights to current target weights.

    The first rebalance buys the whole portfolio, so turnover is 1.0 instead of
    NaN. This keeps the backtest net-of-cost from quietly skipping startup cost.
    """
    if curr.empty:
        return 0.0
    if prev is None or prev.empty:
        return float(curr.abs().sum())
    names = prev.index.union(curr.index)
    return float((curr.reindex(names, fill_value=0.0) - prev.reindex(names, fill_value=0.0)).abs().sum() / 2.0)


def portfolio_return(next_rets: pd.Series, weights: pd.Series) -> float:
    """Portfolio return with missing selected names skipped and reweighted."""
    aligned = next_rets.reindex(weights.index).dropna()
    if aligned.empty:
        return float("nan")
    live_weights = equal_weights(aligned.index)
    return float(aligned.dot(live_weights))


def form_decile_portfolios(
    signal_df: pd.DataFrame,
    return_df: pd.DataFrame,
    decile: float = 0.1,
    min_names: int = 50,
) -> pd.DataFrame:
    """Form top/bottom-decile equal-weight portfolios with weight turnover."""
    if not 0 < decile <= 0.5:
        raise ValueError("decile must be in (0, 0.5]")
    common_dates = signal_df.index.intersection(return_df.index)
    common_tickers = signal_df.columns.intersection(return_df.columns)
    signal_df = signal_df.loc[common_dates, common_tickers]
    return_df = return_df.loc[common_dates, common_tickers]

    rows: list[dict[str, object]] = []
    rebalance_dates: list[pd.Timestamp] = []
    prev_long: pd.Series | None = None
    prev_short: pd.Series | None = None

    for i in range(len(common_dates) - 1):
        date = common_dates[i]
        next_date = common_dates[i + 1]
        scores = signal_df.loc[date].dropna()
        if len(scores) < min_names:
            continue

        n_side = max(int(len(scores) * decile), 1)
        ranked = scores.sort_values(ascending=False)
        long_weights = equal_weights(ranked.head(n_side).index)
        short_weights = equal_weights(ranked.tail(n_side).index)
        next_rets = return_df.loc[next_date]

        long_ret = portfolio_return(next_rets, long_weights)
        short_ret = portfolio_return(next_rets, short_weights)
        rows.append(
            {
                "long": long_ret,
                "short": short_ret,
                "ls": long_ret - short_ret,
                "long_holdings": list(long_weights.index),
                "short_holdings": list(short_weights.index),
                "long_turnover": turnover_from_weights(prev_long, long_weights),
                "short_turnover": turnover_from_weights(prev_short, short_weights),
            }
        )
        rebalance_dates.append(next_date)
        prev_long = long_weights
        prev_short = short_weights

    out = pd.DataFrame(rows, index=pd.Index(rebalance_dates))
    if not out.empty:
        out["ls_turnover"] = out["long_turnover"] + out["short_turnover"]
    return out


def decile_long_returns(signal_df: pd.DataFrame, ret_df: pd.DataFrame, decile: float = 0.1) -> pd.Series:
    """Top-decile equal-weight long-only monthly returns (signal at t, return at t+1).

    Vectorized for speed — the webapp's live stress-test button calls this once
    per click. Ranks cross-sectionally per month, takes the top decile, and
    averages next-month returns. (Semantically equivalent to the per-month loop
    in ``form_decile_portfolios`` but without computing holdings/turnover.)
    """
    ranks = signal_df.rank(axis=1, pct=True)               # NaN stays NaN
    rvals = ranks.values[:-1]                               # signal at t   (T-1, N)
    next_ret = ret_df.values[1:]                            # return at t+1 (T-1, N)
    with np.errstate(invalid="ignore", all="ignore"):
        thr = np.nanquantile(rvals, 1 - decile, axis=1)     # per-row (1-decile) quantile of valid scores
    top = (rvals >= thr[:, None]) & ~np.isnan(rvals)
    valid = (~np.isnan(rvals)).sum(axis=1)
    contrib = np.where(top, next_ret, np.nan)
    with np.errstate(invalid="ignore"):
        long_ret = np.nanmean(contrib, axis=1)
    long_ret = np.where(valid >= 50, long_ret, np.nan)
    return pd.Series(long_ret, index=np.arange(1, signal_df.shape[0]), dtype=float)


def _is_datetime_like(index: pd.Index) -> bool:
    return isinstance(index, pd.PeriodIndex) or pd.api.types.is_datetime64_any_dtype(index)


def _period_index(index: pd.Index) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("M")
    return pd.DatetimeIndex(index).to_period("M")


def align_ff_frame(returns: pd.Series, ff_df: pd.DataFrame) -> pd.DataFrame:
    """Align returns and FF factors by month when dated, otherwise by index."""
    missing = [c for c in FF_COLUMNS if c not in ff_df.columns]
    if missing:
        raise ValueError(f"FF factor frame missing columns: {missing}")

    ret = returns.rename("r").dropna()
    ff = ff_df[FF_COLUMNS].copy()
    if _is_datetime_like(ret.index) and _is_datetime_like(ff.index):
        ret_pm = ret.copy()
        ret_pm.index = _period_index(ret_pm.index)
        ff.index = _period_index(ff.index)
        return ret_pm.to_frame().join(ff, how="inner").dropna()
    return pd.concat([ret, ff], axis=1).dropna()


def fama_french_alpha(long_ret: pd.Series, ff_df: pd.DataFrame, min_obs: int = 20) -> tuple[float, float, float]:
    """Annualized FF 4-factor alpha, alpha t-stat, and regression R^2."""
    model = fama_french_regression(long_ret, ff_df, min_obs=min_obs)
    return float(model.params[0] * 12.0), float(model.tvalues[0]), float(model.rsquared)


def fama_french_regression(long_ret: pd.Series, ff_df: pd.DataFrame, min_obs: int = 20):
    """Fit monthly return on FF 4 factors after subtracting RF."""
    reg = align_ff_frame(long_ret, ff_df)
    if len(reg) < min_obs:
        raise ValueError(f"Need at least {min_obs} aligned observations, got {len(reg)}")
    y = reg["r"] - reg["RF"]
    X = sm.add_constant(reg[FF_FACTOR_COLUMNS], has_constant="add")
    return sm.OLS(y.values, X.values).fit()


def series_metrics(r: pd.Series, freq: int = 12) -> dict[str, float]:
    """Common annualized performance metrics for a monthly return series."""
    r = r.dropna()
    if r.empty:
        return {
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
        }
    ann_return = float(r.mean() * freq)
    ann_vol = float(r.std() * np.sqrt(freq))
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")
    downside = r[r < 0]
    dvol = float(downside.std() * np.sqrt(freq)) if len(downside) > 1 else float("nan")
    sortino = ann_return / dvol if dvol and dvol > 0 else float("nan")
    wealth = (1 + r).cumprod()
    dd = (wealth - wealth.cummax()) / wealth.cummax()
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()),
    }


def marchenko_pastur(eigvals: np.ndarray, n_obs: int, n_assets: int) -> dict[str, float | int]:
    """Marchenko-Pastur bounds and signal eigenvalue count."""
    if n_obs <= 0 or n_assets <= 0:
        raise ValueError("n_obs and n_assets must be positive")
    q = n_obs / n_assets
    sigma2 = float(np.sum(eigvals) / n_assets)
    lam_plus = sigma2 * (1 + 1 / q + 2 * np.sqrt(1 / q))
    lam_minus = sigma2 * (1 + 1 / q - 2 * np.sqrt(1 / q))
    return {
        "q": float(q),
        "sigma2": sigma2,
        "lam_minus": float(lam_minus),
        "lam_plus": float(lam_plus),
        "signal_count": int((eigvals > lam_plus).sum()),
    }


def validate_artifacts(repo_root: Path, required: Mapping[str, ArtifactSpec]) -> None:
    """Check that required notebook outputs exist and have expected columns."""
    missing = [str(spec.path.relative_to(repo_root)) for spec in required.values() if not spec.path.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise ArtifactError(
            "Missing notebook-generated data artifacts. Run notebooks 01 -> 02 -> 03 -> 04 -> 05 -> 06 first:\n"
            f"  - {joined}"
        )

    bad: list[str] = []
    for name, spec in required.items():
        if not spec.columns:
            continue
        try:
            cols = pd.read_csv(spec.path, nrows=0).columns
        except Exception as exc:  # pragma: no cover - surfaced in message
            bad.append(f"{name}: could not read CSV header ({exc})")
            continue
        missing_cols = [c for c in spec.columns if c not in cols]
        if missing_cols:
            bad.append(f"{name}: missing columns {missing_cols}")
    if bad:
        raise ArtifactError("Malformed notebook-generated data artifacts:\n  - " + "\n  - ".join(bad))
