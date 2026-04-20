# Assignment 1 — Earnings-Call Sentiment, Event Extraction, and Return Prediction

**NLP for Finance — Spring 2026**

## Overview

Pipeline that extracts structured sentiment and event data from 131 earnings-call transcripts (14 tickers, ~9 quarters each) and builds a return-prediction backtest.

## Requirements

```
pip install pandas numpy yfinance tqdm requests scikit-learn matplotlib
```

Ollama must be running locally with `qwen3:8b` pulled:

```bash
ollama serve          # in one terminal
ollama pull qwen3:8b  # one-time download (~5 GB)
```

## File Layout

```
Transcript Assignment/
├── Assignment_1_Starter.ipynb   # main notebook
├── ECT/                          # 131 .txt transcripts
├── cache/
│   ├── extractions/              # pre-computed LLM JSON outputs (131 files)
│   └── prices/                   # yfinance parquet cache (15 files)
└── README.md
```

## How to Run

Open `Assignment_1_Starter.ipynb` and run cells top-to-bottom.

| Cell | What it does |
|------|--------------|
| 2 | Imports |
| 4 | Paths & constants |
| 6 | Parse all 131 transcripts |
| 8 | Fetch prices via yfinance (uses disk cache) |
| 10 | LLM extraction functions — skips to cached JSON if present |
| 12 | Build feature DataFrame + QoQ delta features |
| 14 | Train/test split (first 5 calls per ticker = train) |
| 16 | Baseline signal + backtest functions |
| 17 | Equity curve plot |
| 19 | Better model (LogReg + contrarian) |

**The LLM extraction step is already cached.** Cell 10 will load from `cache/extractions/` without calling Ollama. To re-run extraction from scratch, call `extract_one(t, force=True)` or delete the cache.

## Pipeline Summary

1. **Extraction (Task 1):** `qwen3:8b` via Ollama, zero-shot JSON prompt. Each transcript → `{overall_sentiment, sentiment_bucket, wins, risks, guidance, themes}`. 131/131 extracted, 0 parse failures.

2. **Features (Task 2):** QoQ sentiment delta, guidance trajectory streak, risk persistence (word-overlap), n_wins/n_risks deltas.

3. **Model (Task 3):** Logistic regression (6 features, C=0.1) on 21d excess-return sign. Best performer: contrarian variant (flip LR predictions), IC=+0.382.

4. **Backtest (Task 4):** Train = first 5 calls/ticker (~70 obs), test = rest (~61 obs). Entry T+1 close, 21d hold, excess vs. SPY.

## Key Results

| Model | Hit% | Rank IC | Sharpe |
|-------|------|---------|--------|
| Baseline (sent > 0) | 50.9% | NaN | +0.03 |
| LogReg 6-feat | 39.6% | −0.382 | −0.96 |
| Guidance-only | 22.6% | −0.357 | −0.96 |
| **Contrarian LR** | **60.4%** | **+0.382** | **+0.96** |

See `backtest_equity_curve.png` for the cumulative PnL chart.

## Known Limitations

- n=53 test observations after NaN removal — all metrics have wide confidence intervals.
- Raw sentiment is anchored high (~0.85) for most tickers; signal variance is low.
- Contrarian LR was found by flipping a failing model — treat as an observation, not a validated strategy.
- PLTR has no Q&A pairs (pre-recorded format); excluded from Q&A-specific features.
