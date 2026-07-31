from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frd.research import (
    ArtifactError,
    ArtifactSpec,
    block_indices,
    fama_french_alpha,
    form_decile_portfolios,
    momentum_signal,
    validate_artifacts,
)


def test_momentum_signal_uses_prior_eleven_months():
    dates = pd.date_range("2020-01-31", periods=13, freq="ME")
    returns = pd.DataFrame({"AAA": np.ones(13) * 0.01}, index=dates)

    signal = momentum_signal(returns)

    assert pd.isna(signal.iloc[10, 0])
    assert signal.iloc[11, 0] == pytest.approx(0.11)
    assert signal.iloc[12, 0] == pytest.approx(0.11)


def test_decile_portfolios_include_initial_cost_and_weight_turnover():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    signal = pd.DataFrame(
        {
            "A": [4.0, 4.0, 4.0],
            "B": [3.0, 1.0, 1.0],
            "C": [2.0, 3.0, 3.0],
            "D": [1.0, 2.0, 2.0],
        },
        index=dates,
    )
    returns = pd.DataFrame(
        {
            "A": [0.00, 0.10, 0.20],
            "B": [0.00, 0.00, 0.00],
            "C": [0.00, 0.04, 0.06],
            "D": [0.00, 0.00, 0.00],
        },
        index=dates,
    )

    port = form_decile_portfolios(signal, returns, decile=0.5, min_names=1)

    assert port.index.tolist() == list(dates[1:])
    assert port["long"].iloc[0] == pytest.approx(0.05)
    assert port["long"].iloc[1] == pytest.approx(0.13)
    assert port["long_turnover"].iloc[0] == pytest.approx(1.0)
    assert port["long_turnover"].iloc[1] == pytest.approx(0.5)
    assert port["ls_turnover"].iloc[0] == pytest.approx(2.0)


def test_fama_french_alpha_aligns_month_start_and_month_end():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    ff_dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    returns = pd.Series(0.01 + np.tile([-0.001, 0.001], 12), index=dates)
    ff = pd.DataFrame(
        {
            "Mkt-RF": np.zeros(24),
            "SMB": np.zeros(24),
            "HML": np.zeros(24),
            "Mom": np.zeros(24),
            "RF": np.zeros(24),
        },
        index=ff_dates,
    )

    alpha, tstat, r2 = fama_french_alpha(returns, ff)

    assert alpha == pytest.approx(0.12)
    assert np.isfinite(tstat)
    assert r2 >= 0


def test_block_indices_are_in_bounds_and_requested_length():
    rng = np.random.RandomState(3)
    idx = block_indices(25, 5, rng)

    assert len(idx) == 25
    assert idx.min() >= 0
    assert idx.max() < 25


def test_validate_artifacts_reports_missing_and_malformed(tmp_path: Path):
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    good.write_text("a,b\n1,2\n", encoding="utf-8")
    bad.write_text("a\n1\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="Missing notebook-generated"):
        validate_artifacts(
            tmp_path,
            {
                "good": ArtifactSpec(good, ("a", "b")),
                "missing": ArtifactSpec(tmp_path / "missing.csv", ("x",)),
            },
        )

    with pytest.raises(ArtifactError, match="Malformed notebook-generated"):
        validate_artifacts(tmp_path, {"bad": ArtifactSpec(bad, ("a", "b"))})
