# Factor Risk Decomposition

*A quantitative pipeline, viewed through the lens of linear algebra.*

An end-to-end quant research pipeline — from raw price data to a backtested long-only momentum portfolio with a Fama–French alpha, explicit risk decomposition, and synthetic stress testing. The target reader is someone fluent in linear algebra who wants to see how those tools show up in finance. Every concept is introduced first via its linear-algebra structure (matrices, vectors, projections, eigendecompositions, subspaces) and then named in finance terms.

## Contents

1. [Finance ↔ Linear Algebra Dictionary](#finance--linear-algebra-dictionary)
2. [Setup and Reproducibility](#setup-and-reproducibility)
3. [Part I — Data and Factor Analysis (Notebooks 01–03)](#part-i--data-and-factor-analysis-notebooks-0103)
4. [Part II — Backtest and Risk Decomposition (Notebooks 04–05)](#part-ii--backtest-and-risk-decomposition-notebooks-0405)
5. [Part III — Synthetic Markets and Stress Testing (Notebook 06)](#part-iii--synthetic-markets-and-stress-testing-notebook-06)
6. [Results Summary](#results-summary)
7. [Limitations](#limitations)
8. [Webapp](#webapp)
9. [Tech Stack](#tech-stack)


## Finance $\leftrightarrow$ Linear Algebra Dictionary

The single most useful thing to keep in mind: **a stock return panel is a matrix**, and almost every operation in quantitative finance is a standard linear-algebra operation on that matrix — a projection, an inner product, a quadratic form, or an eigendecomposition.

Each notebook opens with its own "Terms used in this notebook" table covering only what appears there. The project-wide reference, with the notebook(s) where each term appears, is below.

Notebook numbering: **01** Data & market stats · **02** Factor diagnostics · **03** Signal construction · **04** Backtest & performance · **05** Risk decomposition (PCA) · **06** Synthetic markets & stress testing.

### 1. The data object

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Return** | Fractional price change: $r_{t} = p_t / p_{t-1} - 1$ | 01 |
| **Return panel** $\mathbf{R}$ | Matrix $\mathbf{R}\in\mathbb{R}^{T\times N}$: months (rows) $\times$ stocks (columns); sparse (NaN where a stock didn't trade yet) | 01–06 |
| **Cross-section** | A row $r_t \in \mathbb{R}^{N_t}$ — all stock returns at one date $t$ | 01, 02, 03 |
| **One stock's history** | A single column of $\mathbf{R}$ | 01 |
| **Universe** | The set of columns we're allowed to hold — here the S&P 500 large-caps | 01 |
| **Sector** | A categorical partition of the column index set (Technology, Financials, …); one-hot form $\rightarrow$ indicator matrix $\mathbf{D}$ | 01, 03 |
| **Dispersion** | Cross-sectional std dev — the spread of returns across a row of $\mathbf{R}$ | 01, 06 |
| **Equal-weight index** | Mean of a row: $\bar{r}_t = \tfrac{1}{N_t}\mathbf{1}^\top r_t$ (portfolio with $w_i = 1/N_t$) | 01, 04 |
| **Survivorship bias** | Columns are a *non-random* sample: delisted/bankrupt names are missing, biasing returns upward | 01, 04 |

### 2. Factors and signals

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Factor / signal** | Vector $f_t \in \mathbb{R}^{N_t}$ — one score per stock; we ask whether $f_t$ predicts $r_{t+1}$ | 02, 03, 04 |
| **Momentum (12-1)** | Trailing 12-month return skipping the last month — the headline signal; persistent drift | 02, 03, 04 |
| **Value / Quality / Low-vol** | Other candidate factor vectors (proxied by price data here) | 02, 03 |
| **Information coefficient (IC)** | Cosine similarity $\cos\theta = \dfrac{f_t^\top r_{t+1}}{\|f_t\|\ \|r_{t+1}\|}$ (ranks $\rightarrow$ Spearman IC) | 02, 03 |
| **Information ratio (IR)** | Mean IC / std(IC) — a signal-to-noise ratio for the factor | 02, 04 |
| **Rank** | A permutation of $\{1,\dots,N\}$; makes factors comparable and robust to outliers | 02 |
| **Walk-forward validation** | Split the sample into sub-windows and test stability of IC / performance across regimes | 02, 04 |
| **Winsorize** | Clip extreme entries of $f$ at $\pm 3\sigma$ to tame outliers | 03 |
| **Z-score** | Center each cross-section to mean 0, std 1 | 03 |
| **Sector neutralization** | Orthogonal projection $f_\perp = (I - P)f$ with $P = \mathbf{D}(\mathbf{D}^\top\mathbf{D})^{-1}\mathbf{D}^\top$ onto sector dummies | 03, 04 |
| **Composite signal** | A linear combination $c_t = \sum_k w_k f_{k,\perp}$ of neutralized factor vectors | 03 |

### 3. Portfolios and performance

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Portfolio weights** $w$ | A vector; long-only: $w \ge 0,\ \sum w_i = 1$; long-short: $\sum w_i = 0$ | 04, 05 |
| **Portfolio return** | Inner product $w^\top r_{t+1}$ | 04 |
| **Portfolio variance** | Quadratic form $w^\top \Sigma w$ | 05 |
| **Turnover** | One-way turnover $\tfrac{1}{2}\|w_t - w_{t-1}\|_1$ — how much the weight vector changes between rebalances | 04 |
| **Transaction cost** | $c \cdot \tfrac{1}{2}\|w_t - w_{t-1}\|_1$ — proportional to one-way turnover | 04 |
| **Backtest** | Replay history: form $w_t$ each month, accumulate $w_t^\top r_{t+1}$ net of costs | 04 |
| **Sharpe / Sortino / Calmar** | Signal-to-noise ratios on portfolio returns (downside-only for Sortino; return/max-DD for Calmar) | 04 |
| **Max drawdown** | Largest peak-to-trough drop of the equity curve | 04 |
| **Active return / IR** | Portfolio return minus benchmark; IR = active return / tracking error | 04 |
| **Beta** | Projection length of a return onto a factor: $\beta_i = \mathrm{Cov}(r_i, f) / \mathrm{Var}(f)$ | 04, 05 |
| **Alpha** | Orthogonal residual of portfolio returns after projecting onto the factor span | 04 |
| **Fama–French regression** | OLS projection of returns onto a factor basis; alpha = intercept, beta = coordinates | 04 |

### 4. Risk decomposition

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Covariance matrix** $\Sigma$ | $\Sigma = \tfrac{1}{T-1}\mathbf{X}_c^\top \mathbf{X}_c$ — input to every risk calculation | 05 |
| **PCA / eigendecomposition** | $\Sigma = V\Lambda V^\top$; eigenvectors = principal components, eigenvalues = explained variance | 05 |
| **Scree plot** | Eigenvalues in decreasing order — visualizes the spectrum | 05 |
| **Marchenko–Pastur cutoff** | Upper edge $\lambda_+$ of the noise spectrum of a random covariance matrix; eigenvalues above it = signal | 05 |
| **Ledoit–Wolf shrinkage** | $\hat\Sigma = \delta F + (1-\delta)S$ — convex combination of sample $S$ and a structured target $F$ | 05 |
| **Systematic / idiosyncratic risk** | Variance in the top-$k$ factor subspace vs. its orthogonal complement | 05 |

### 5. Synthetic markets & stress testing (notebook 06)

| Term | Linear-algebra meaning | Notebooks |
|------|------------------------|:---------:|
| **Truncated SVD / factor model** | $\mathbf{R} \approx \mathbf{F}\mathbf{B}^\top + \mathbf{E}$ — best rank-$k$ approximation (Eckart–Young) | 06 |
| **Block bootstrap** | Resample rows of $\mathbf{F}$ in blocks to preserve temporal correlation | 06 |
| **Conditional VAE** | Learn a latent $\mathbf{z} \sim \mathcal{N}(\boldsymbol\mu, \mathrm{diag}(\boldsymbol\sigma^2))$ for factor returns (autoregressive) | 06 |
| **Stress test / synthetic market** | Re-run the backtest on generated $\mathbf{R}_{\text{synth}}$ to get a distribution of alpha | 06 |

---

## Overview

This project constructs and backtests a **sector-neutralized momentum factor** about ~500 US large-cap stocks (2005–2025) using only free data. The pipeline:

> 1. **Assemble** a current-constituent, survivorship-biased universe and return panel (the matrix $\mathbf{R}$) from free/cached price data
> 2. **Diagnose** four candidate factors via information coefficient analysis and walk-forward subperiod stability
> 3. **Select** momentum as the headline factor (the only one with positive IC) and sector-neutralize it via orthogonal projection
> 4. **Backtest** a monthly rebalanced top-decile long-only portfolio with transaction costs and walk-forward validation
> 5. **Decompose** portfolio risk into systematic vs. idiosyncratic components via PCA (eigendecomposition + random matrix theory)
> 6. **Stress test** (notebook 06) by generating synthetic markets and re-running the backtest across alternative histories

The project is structured in three parts:

* **Part I — Data and Factor Analysis** (notebooks 01–03)
* **Part II — Backtest and Risk Decomposition** (notebooks 04–05)
* **Part III — Synthetic Markets and Stress Testing** (notebook 06)

---

## Project Structure

```text
Factor-Risk-Decomposition/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # S&P 500 constituents, monthly prices, Fama–French factors
│   └── processed/            # cleaned panels, factor exposures, signals, backtest & risk outputs
├── images/                   # notebook outputs (gitignored)
│   ├── 01_market_stats/
│   ├── 02_factor_diagnostics/
│   ├── 03_factor_construction/
│   ├── 04_backtest/
│   ├── 05_risk_decomposition/
│   └── 06_synthetic_markets/
└── notebooks/
    ├── 01_data_overview_and_market_stats.ipynb
    ├── 02_factor_analysis_and_diagnostics.ipynb
    ├── 03_factor_construction_and_composite_signal.ipynb
    ├── 04_backtest_and_performance.ipynb
    ├── 05_risk_decomposition_via_PCA.ipynb
    └── 06_synthetic_market_generation.ipynb
```

---

## Setup and Reproducibility

### Requirements

```bash
pip install -r requirements.txt
```

For the dashboard-only runtime, install the smaller pinned set:

```bash
pip install -r requirements-webapp.txt
```

### Data Sources (all free)

- **Prices:** adjusted close via `yfinance` (2005–2025)
- **Constituents:** cached S&P 500 snapshot in `data/raw/constituents.csv`; set `REFRESH_DATA=true` before notebook 01 to intentionally replace it from Wikipedia
- **Benchmark factors** (notebook 04): Kenneth French Data Library (MKT, SMB, HML, MOM, RF) via `pandas-datareader` — the standard "Fama–French" factors used to decompose returns into market, size, value, and momentum components

No paid data feed is required to run the pipeline end-to-end.

### Running the project

Run notebooks in order:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06
```

Each notebook is self-contained and writes to `data/processed/` and `images/`, so later notebooks pick up where earlier ones left off. Random state is fixed at `RANDOM_STATE = 3` throughout. Generated CSVs are intentionally gitignored; rerun the notebooks to refresh them, and use `REFRESH_DATA=true` only when you want a new constituent snapshot.

---

## Part I — Data and Factor Analysis (Notebooks 01–03)

This part builds the data matrix $\mathbf{R}$, diagnoses individual factor vectors, and selects the best factor with sector neutralization (orthogonal projection).

### 1. Data Overview and Market Statistics

We use a cached snapshot of S&P 500 constituents and download adjusted close prices. Because that snapshot is still based on a modern S&P 500 membership list, it introduces **survivorship bias** — names that went bankrupt or were delisted between 2005 and today won't appear. In linear-algebra terms: the columns of $\mathbf{R}$ are a non-random subset of all stocks that existed; the columns we *don't* see are exactly the ones that went to zero or got delisted, biasing returns upward. Notebook 04 includes a sensitivity analysis for this.

Key findings:
* **Universe breadth** rises from ~385 to ~501 stocks over the sample — but the column set is fixed to *today's* constituents, so this counts how many of today's survivors had price data in month $t$. The matrix isn't truly "getting wider"; its survivor-only columns fill in over time.
* Average monthly cross-sectional return dispersion ~7.5% (the spread across a row of $\mathbf{R}$).
* Equal-weight universe Sharpe (rf=0) $\approx$ 0.98 over the full period.
* Sector composition spans 11 GICS sectors, dominated by Technology, Financials, and Healthcare.

### 2. Factor Analysis and Diagnostics

A **factor** is a vector $f_t \in \mathbb{R}^{N_t}$ — one score per stock at each date. We compute four standard factors, all as cross-sectional ranks at each month-end:

| Factor | Definition | Finance intuition |
|--------|-----------|-------|
| Momentum (12-1) | Trailing 12-month return excluding most recent month | Stocks that went up tend to keep going up; skip month avoids reversal contamination |
| Value (proxy) | Inverse 60-month momentum | Cheap stocks (low long-term return) may mean-revert; real impl would use book-to-price |
| Quality (proxy) | 12-month Sharpe-like ratio | Stocks with smooth positive returns are "higher quality"; real impl would use ROE/gross profitability |
| Low volatility | Inverse 60-month trailing volatility | Low-risk stocks tend to outperform on a risk-adjusted basis |

The **information coefficient (IC)** is the Spearman rank correlation (cosine similarity of rank vectors) between $f_t$ and $r_{t+1}$. We also run **walk-forward subperiod IC analysis** over 5-year windows.

Key findings (full-sample monthly IC):
* **Momentum is the only usable signal in this setup** — mean IC +0.006, IR 0.11. That is weak in absolute terms; the point is not that this is an industry-grade factor, but that it is the only price-based proxy here worth carrying forward. Value (-0.022), quality (-0.003, ~zero), and low-vol (-0.026) are all negative or flat.
* **Walk-forward:** momentum's IC is positive in **2 of 4** five-year windows (2011–16 and 2021–26); negative in 2006–11 and 2016–21. The signal is real but regime-dependent.
* **IC decay:** momentum's edge fades beyond 1 month (negative at 3, 6, 12-month horizons).
* **Turnover:** momentum rank autocorrelation ~0.89 (the vector rotates meaningfully each month).
* **Cross-factor correlation:** momentum and quality are 0.86 correlated — nearly collinear, redundant information.

### 3. Factor Construction and Signal Selection

Notebook 02 established momentum as the only factor with positive IC. This notebook:

1. **Winsorizes** raw factor exposures (clip extreme entries at $\pm 3 \sigma$)
2. **Z-scores** each factor cross-sectionally (center to mean 0, std 1)
3. **Sector-neutralizes** via orthogonal projection: $f_\perp = (\mathbf{I} - \mathbf{P})f$ with $\mathbf{P} = \mathbf{D}(\mathbf{D}^\top\mathbf{D})^{-1}\mathbf{D}^\top$
4. **Compares** momentum-only vs. a 4-factor equal-weight composite

Sector neutralization slightly improves momentum (IR 0.11 raw $\rightarrow$ 0.15 neutralized). Combining momentum with two negative-IC factors (value, lowvol) and one near-zero factor (quality) **dilutes** the signal — the 4-factor composite has negative IC. This is the linear-algebra intuition made empirical: adding vectors that point in the wrong direction moves the sum away from the target.

**The headline signal is momentum-only, sector-neutralized** $\rightarrow$ saved as `momentum_signal.csv`. This is what the backtest trades.

---

## Part II — Backtest and Risk Decomposition (Notebooks 04–05)

This part turns the signal vector into a portfolio (a weight vector), measures performance (alpha as an orthogonal residual), and decomposes portfolio risk (a quadratic form) via eigendecomposition.

### 4. Backtest and Performance

A portfolio is a weight vector $w$. We form a top-decile long-only portfolio at each monthly rebalance:

* **Signal at month-end $t$, traded at $t+1$** to avoid look-ahead bias
* **Portfolio return** = $w^\top r_{t+1}$
* **Transaction costs:** 5 bps per unit of one-way turnover, $c \cdot \tfrac{1}{2}\|w_t - w_{t-1}\|_1$
* **Benchmark:** the **equal-weight (EW) universe** — since the portfolio is equal-weighted within the decile, the fair comparison is an equal-weight portfolio of *all* stocks, isolating stock-picking from the size effect.

**Fama–French alpha:** a regression of portfolio excess returns onto MKT/SMB/HML/MOM; the **alpha** is the intercept — average return not explained by exposure to those known factors. Notebook 04 reports both ordinary OLS t-stats and HAC/Newey-West t-stats.

| Portfolio | Ann. Return | Sharpe | Max DD | Sortino |
|-----------|------------:|-------:|-------:|--------:|
| Long-Only (net) | 19.5% | 1.00 | -57% | 1.40 |
| EW Universe | 15.9% | 0.95 | -47% | 1.27 |
| Long-Short (net) | -0.7% | -0.04 | -70% | — |

| Portfolio | FF 4-factor alpha (ann.) | OLS t-stat | HAC t-stat | MKT $\beta$ | MOM $\beta$ |
|-----------|-------------------------:|-----------:|-----------:|------:|------:|
| **Long-Only** | **+5.97%** | **3.98** | **4.17** | 1.19 | 0.25 |
| Long-Short | -2.95% | -1.41 | -1.37 | 0.14 | 0.91 |

The long-only portfolio beats the EW universe by **3.5% per year**, though that raw active edge is only marginal (IR 0.42, t = 1.89). After beta matching, the raw outperformance versus the equal-weight universe falls to about **0.5% per year**, so the factor-adjusted alpha is the stronger result. The long-short alpha is not significant — the short side adds noise, so momentum's predictive power is concentrated on the long side in this universe.

**Walk-forward (5-year windows):** long-only Sharpe is positive in **4 of 4** windows; the active return (vs EW universe) is positive in **3 of 4**. The exception is 2006–2011 (active -4.8%), which spans the 2008–09 momentum crash — a well-documented regime where momentum reverses. Per-window information ratios: -0.51, 0.71, 0.82, 1.08, improving over the sample.

**Survivorship-bias sensitivity:** re-running the Fama–French regression with synthetic return drag, the alpha survives **2%** annual drag under both flat and crash-concentrated assumptions. At **3%**, the flat-drag test is borderline, while the crash-concentrated version loses significance.

**Pipeline robustness:** notebook 04 now checks decile cutoffs of 5%, 10%, 15%, and 20%, plus 1-, 2-, and 3-month rebalance intervals. Across that grid, alpha remains positive and HAC-significant. This helps with parameter fragility, but does not solve the larger universe-construction limitation.

### 5. Risk Decomposition via PCA

Portfolio risk is the quadratic form $w^\top \Sigma w$. This notebook decomposes it using the eigendecomposition of $\Sigma$.

* **PCA via scikit-learn** (`PCA` from `sklearn.decomposition`), fit on the centered return matrix (a balanced panel of 394 stocks over 240 months). The linear-algebra content is the same eigendecomposition $\Sigma = V\Lambda V^\top$.
* **Marchenko–Pastur cutoff** (random matrix theory): eigenvalues of a random covariance matrix fall inside $[\lambda_-, \lambda_+]$; eigenvalues above $\lambda_+$ are statistically significant factors. Here **9 eigenvalues** exceed the cutoff. PC1 explains 34% of variance and is essentially the market factor (correlation 0.96 with MKT-RF); PC2 loads on value (HML, 0.65).
* **Ledoit–Wolf shrinkage** ($\hat\Sigma = \delta F + (1-\delta)S$) regularizes the noisy sample covariance; out-of-sample it lowers the Frobenius estimation error by ~5%.
* **Variance decomposition:** splitting $\Sigma$ into the top-$k$ factor subspace and its orthogonal complement,

$$w^\top \Sigma w = \underbrace{w^\top \mathbf{B} \Sigma_f \mathbf{B}^\top w}_{\text{systematic}} + \underbrace{w^\top (\Sigma - \mathbf{B}\Sigma_f \mathbf{B}^\top) w}_{\text{idiosyncratic}}.$$

The momentum long-only portfolio's risk is **~90.7% systematic** and **~9.3% idiosyncratic** — overwhelmingly driven by common factor exposures, consistent with a diversified ~50-stock top-decile portfolio. One caveat: the Fama–French alpha is orthogonal to the Fama–French benchmark factors, not literally to the PCA basis. These are related decompositions, but they are not the same coordinate system.

---

## Part III — Synthetic Markets and Stress Testing (Notebook 06)

A single backtest is one draw from a distribution of possible histories. This part asks whether the alpha is unusually dependent on the specific historical ordering of months. We generate many synthetic markets from the notebook 05 factor model and re-run the momentum backtest on each.

### 6. Synthetic Market Generation

Reusing the truncated-SVD factor model $\mathbf{R} \approx \mathbf{F}\mathbf{B}^\top + \mathbf{E}$ (top-$k$ eigenvectors $\mathbf{B}$, factor scores $\mathbf{F}$, residuals $\mathbf{E}$, with $k$ estimated by Marchenko–Pastur), we generate alternative histories and re-derive the full momentum pipeline (signal $\rightarrow$ decile portfolio $\rightarrow$ Fama–French regression) on each.

* **Block bootstrap — the trustworthy generator.** Resample time indices in blocks (length $\approx\sqrt{T}$) and reconstruct $\mathbf{R}_\text{synth}[t]=\mathbf{F}[\text{idx}_t]\mathbf{B}^\top+\mathbf{E}[\text{idx}_t]+\bar r$, using the **same** index for factors, residuals, *and* the Fama–French factors — so each synthetic timeline is a reshuffling of real joint return rows. Over 300 paths, the mean synthetic alpha $\approx$ 4.85%/yr and **~56% of paths beat the real 4.49%**. The real alpha sits near the median: it is **typical of the factor structure, not a lucky sequence**.
* **Conditional VAE — a cautionary result.** An autoregressive VAE on $\mathbf{F}$ ($f_{t-1}\to(\mu,\sigma)\to z\to\hat f_t$) can generate factor paths, but reconstructing markets from a *generated* $\hat{\mathbf{F}}$ stitched to independently-resampled residuals **fabricates** return rows with spurious cross-sectional persistence — inflating momentum alphas to 10–25%. The lesson: a generative model that splits $\mathbf{R}=\mathbf{F}\mathbf{B}^\top+\mathbf{E}$ and regenerates the parts separately can inject the very signal under test, so we do **not** rely on it.

**Honest scope.** Both generators hold $\mathbf{B}$ fixed and preserve the factor structure that *produces* the edge, so this tests **path dependence**, not "does momentum work without a momentum factor." Combined with notebook 04's walk-forward checks, HAC alpha, survivorship sensitivity, and robustness grid, the evidence is stronger than a single backtest — with the residual caveat that the bootstrap cannot rule out an unmodeled structural explanation.

---

## Results Summary

| Metric | Long-Only (net) | EW Universe | Long-Short (net) |
|--------|-----------------|-------------|------------------|
| Annualized return [mean of the inner product $w^\top r_{t+1}$] | 19.5% | 15.9% | -0.8% |
| Sharpe ratio [$\bar r_p / \mathrm{std}(r_p)$ — a signal-to-noise ratio] | 1.01 | 0.95 | -0.04 |
| Max drawdown [largest peak-to-trough drop of the compounded wealth curve] | -57% | -47% | -70% |
| FF 4-factor alpha (annualized) [regression intercept after controlling for FF factors] | **+5.97% (OLS t = 3.98, HAC t = 4.17)** | — | -2.95% (OLS t = -1.41, HAC t = -1.37) |
| Active return vs EW universe [$w^\top r$ minus its projection onto $\mathbf{1}$] | +3.5% (IR 0.42, t = 1.89) | — | — |
| Walk-forward: Sharpe positive [positive signal-to-noise in each sub-window] | 4 of 4 windows | — | — |
| Walk-forward: active positive [positive projection residual in each sub-window] | 3 of 4 windows | — | — |
| Survivorship drag to lose alpha [bias from the non-random column set needed to cancel $\alpha$] | survives 2%; flat 3% borderline, concentrated 3% fails | — | — |
| Systematic risk share [variance in the top-$k$ eigenspace, $w^\top B\Sigma_f B^\top w$, as a share of $w^\top \Sigma w$] | ~90.7% | — | — |
| Stress test (block bootstrap) [share of 300 synthetic markets whose alpha $\ge$ the real alpha] | ~56% beat real $\rightarrow$ typical, not path-dependent | — | — |

**Bottom line:** a sector-neutralized momentum signal, traded long-only, generates a Fama–French 4-factor alpha of **5.97% annualized** (OLS t = 3.98, HAC t = 4.17). The evidence is meaningfully better than a single backtest because it includes walk-forward checks, beta diagnostics, survivorship-drag stress tests, a decile/rebalance robustness grid, and synthetic-market path tests. The long-short variant does not work — the edge is on the long side.

---

## Limitations

* **Survivorship bias.** The universe is reconstructed from a cached modern S&P 500 snapshot, so delisted/bankrupt names are missing. The sensitivity analysis (notebook 04) shows the alpha survives 2% annual return drag under flat and crash-concentrated assumptions; at 3%, the conclusion depends on the drag model. A survivorship-free database (CRSP) would eliminate this concern entirely.
* **Price-based factor proxies.** Value and quality are proxied by price-based measures rather than fundamentals, and have negative/near-zero IC. A real implementation with Compustat/Sharadar fundamentals might produce a working multi-factor composite.
* **Marginal raw active return.** The long-only portfolio beats the EW universe by only 3.5%/yr (t = 1.89); the statistically strong result is the *factor-adjusted* alpha (5.97%, t = 3.98), not the raw active return.
* **No intraday execution modeling.** Transaction costs are a flat 5 bps per unit of one-way turnover. Real slippage depends on order size, liquidity, and volatility.
* **Limited rebalance grid.** Notebook 04 now checks 1-, 2-, and 3-month rebalance intervals, but does not model daily/weekly trading or alternate calendar-day execution.
* **Momentum crash risk.** The 2006–2011 walk-forward window shows negative active return, driven by the 2008–09 momentum crash. A crash-protection overlay (e.g. volatility scaling) would improve robustness.
* **Stress-test scope.** The synthetic-market bootstrap preserves the factor structure (the momentum PC lives in $\mathbf{F}$), so it tests **path dependence**, not "momentum without a momentum factor"; the conditional-VAE generator was found to inflate alphas (it fabricates cross-sectional persistence) and is not relied upon.

---

## Webapp

An interactive dashboard in `webapp/` showcases the pipeline: a **FastAPI** backend + a single-page **Plotly.js** frontend themed to match the rest of the site. It consumes the precomputed CSVs in `data/processed/` and recomputes the light ML **once at startup** (PCA + Marchenko–Pastur cutoff, the Fama–French alpha, and the NB06 factor model for the live button), so the numbers always match the notebooks.

Sections: **Strategy** (the trading rule, realized alpha, and short FF/t-stat explanation), **Generate** (a live block-bootstrap alpha generator), **Performance** (equity curves and drawdowns), **Factors** (IC bars, correlation heatmap, walk-forward), **Risk** (scree + MP cutoff, systematic/idiosyncratic split), **Ticker explorer** (PC1 vs PC2 loadings), and **Process** (the notebook-by-notebook research pipeline).

### Run locally

```bash
pip install -r requirements-webapp.txt
uvicorn webapp.app:app --host 127.0.0.1 --port 8055
# open http://127.0.0.1:8055
```

Or with Docker (binds `127.0.0.1:8055`):

```bash
docker compose -f docker-compose.webapp.yml up --build
```

### Deploy behind Caddy

The proxy compose only `expose`s its port on your existing `caddy` Docker network — no host port, so it coexists with other sites

```bash
docker compose -f docker-compose.webapp.proxy.yml up -d --build
```

Then add a Caddy site block reverse-proxying to the container:

```caddy
frd.example.com {
    reverse_proxy factor-risk-decomposition-webapp:8055
}
```

The data stays **mounted read-only** (`./data:/app/data`). The app validates the required CSV artifacts at startup and tells you to run notebooks `01 -> 06` if anything is missing or malformed.

---

## Tech Stack

Python, pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, yfinance, pandas-datareader, joblib, torch, FastAPI, Uvicorn. See `requirements.txt`, `requirements-webapp.txt`, and `requirements-dev.txt`.

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## License

MIT — see [LICENSE](LICENSE). For educational purposes. Price data belongs to Yahoo Finance / yfinance; factor data belongs to the Kenneth French Data Library.
