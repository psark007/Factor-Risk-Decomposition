"""Shared research utilities for Factor Risk Decomposition."""

from .research import (
    FF_FACTOR_COLUMNS,
    ArtifactError,
    ArtifactSpec,
    block_indices,
    decile_long_returns,
    fama_french_alpha,
    fama_french_regression,
    form_decile_portfolios,
    marchenko_pastur,
    momentum_signal,
    series_metrics,
    validate_artifacts,
)

__all__ = [
    "FF_FACTOR_COLUMNS",
    "ArtifactError",
    "ArtifactSpec",
    "block_indices",
    "decile_long_returns",
    "fama_french_alpha",
    "fama_french_regression",
    "form_decile_portfolios",
    "marchenko_pastur",
    "momentum_signal",
    "series_metrics",
    "validate_artifacts",
]
