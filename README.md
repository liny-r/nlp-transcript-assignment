# Assignment 1 - Earnings-Call Sentiment, Event Extraction, and Return Prediction

NLP for Finance - Spring 2026

## Overview

This repository contains an end-to-end pipeline for 131 earnings-call transcripts (14 US tickers), including:

- transcript parsing
- LLM event/sentiment extraction
- quarter-over-quarter feature engineering
- train/test modeling
- simple backtests and diagnostics

Primary implementation is in `Assignment_1_YueqiLin.ipynb`.

## Requirements

Install Python dependencies:

```bash
pip install pandas numpy yfinance tqdm requests scikit-learn matplotlib pysentiment2
```

Optional (only needed if re-running extraction instead of using cache):

```bash
ollama serve
ollama pull qwen3:8b
```

## Project Layout

```text
Transcript Assignment/
|- Assignment_1_YueqiLin.ipynb
|- Assignment_1_writeup_YueqiLin.pdf
|- ECT/                      # 131 transcript .txt files
|- cache/
|  |- extractions/           # cached LLM JSON extractions (131 files)
|  |- prices/                # cached yfinance parquet files (tickers + SPY)
|- backtest_equity_curve.png
|- baseline_llm_vs_lm.png
`- README.md
```

## Reproduction Steps

1. Open `Assignment_1_YueqiLin.ipynb`.
2. Run all cells from top to bottom.
3. The notebook will load cached extraction JSON from `cache/extractions/` and cached price data from `cache/prices/` when present.
4. If caches are deleted, extraction/price steps will rebuild them.

Notes:

- Train/test split is by time within each ticker (first 5 calls train, remaining calls test).
- Entry convention is T+1 close to avoid look-ahead from after-hours calls.
- The notebook evaluates multiple horizons and model variants in different sections; check each section header for the exact target used.

## What to Submit

For a reproducible submission, include at minimum:

- `Assignment_1_YueqiLin.ipynb`
- `Assignment_1_writeup_YueqiLin.pdf`
- `README.md`
- `cache/extractions/*.json`
- `cache/prices/*`
- generated figures referenced in the writeup

## Known Caveats

- Small out-of-sample size, so metrics are noisy.
- PLTR has no standard Q&A section in this dataset format, so Q&A-derived features are NaN for PLTR calls.
- LLM sentiment can be anchored high; delta and auxiliary features are included to mitigate this.
