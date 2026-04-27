# Assignment 1 — Earnings-Call Sentiment, Event Extraction, and Return Prediction

NLP for Finance — Spring 2026

## Overview

End-to-end pipeline for 131 earnings-call transcripts (14 US tickers, ~9–10 quarters each):

- Transcript parsing (prepared remarks + Q&A separation, speaker labelling)
- LLM event/sentiment extraction via qwen3:8b (Ollama local)
- Quarter-over-quarter feature engineering (17 features)
- Train/test modelling with GridSearchCV (LogReg L1/L2, GBC, Ridge, Lasso)
- Backtest: directional accuracy, rank IC, Sharpe, equity curve vs. SPY

Primary implementation: `Assignment_1_YueqiLin.ipynb`  
Writeup: `Assignment_1_writeup_YueqiLin.md`

---

## Requirements

```bash
pip install pandas numpy yfinance tqdm requests scikit-learn matplotlib pysentiment2 pyarrow
```

LLM extraction (only needed if re-running from scratch — cached output is included):

```bash
ollama serve
ollama pull qwen3:8b
```

> **Note:** set `num_ctx` to at least 49152 in the Ollama payload. The longest transcript
> (JNJ Q4-2024) requires ~45k tokens. The notebook sets this automatically.
>
> **Note:** qwen3:8b has thinking mode enabled by default. The notebook sets `think: false`
> at both the top-level payload and inside `options`. Omitting either causes the model to
> exhaust its token budget on `<think>` blocks and return empty output.

---

## Project Layout

```
Transcript Assignment/
├── Assignment_1_YueqiLin.ipynb       # main notebook
├── Assignment_1_writeup_YueqiLin.md  # PDF writeup source
├── README.md
├── figures/                          # report/backtest figures
│   ├── backtest_equity_curve.png
│   ├── baseline_llm_vs_lm.png
│   ├── lm_vs_llm.png
│   ├── ls_portfolio.png
│   ├── external_signal_comparison.png
│   └── external_signal_coefs.png
├── ECT/                              # 131 transcript .txt files (not included in zip)
└── cache/
    ├── extractions/      # cached LLM JSON extractions (131 files, ~4 MB)
    ├── prices/                       # cached yfinance parquet files (15 tickers + SPY)
    └── lm_sentiment.json             # cached Loughran-McDonald scores (131 entries)
```

---

## Reproduction Steps

**To reproduce model and backtest results from cache (no LLM needed):**

1. Open `Assignment_1_YueqiLin.ipynb`
2. Run all cells top to bottom
3. The notebook loads:
   - LLM extractions from `cache/extractions/`
   - Price data from `cache/prices/`
   - LM sentiment from `cache/lm_sentiment.json`
4. All three caches are included — no re-extraction or API calls required

**To re-run LLM extraction from scratch:**

1. Ensure Ollama is running (`ollama serve`) with qwen3:8b pulled
2. Delete or rename `cache/extractions/`
3. Re-run §4 (LLM Extraction) cells — extraction caches per-transcript as it goes

**To refresh price data:**

- Delete `cache/prices/` — §3 rebuilds it via `yfinance`
- Prices are cached for 24 hours; stale files are auto-refreshed on next run

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM | qwen3:8b (local) | Speed (~1.5 min/transcript on M1 Pro); no API cost |
| Guidance schema | Structured list `[{line, direction}]` | Retains per-line detail lost in scalar "raised/lowered" |
| Horizon | 5-day excess return vs. SPY | Short enough to isolate post-earnings drift; excess removes beta |
| Train/test split | First 5 calls per ticker = train | Time-based split; no future data leaks into training |
| Final model | LogReg C=0.5 L1 (17 input features, 4 active) | Highest test-set IC (+0.151) at n=46; L1 handles small-n overfitting |

---

## Train/Test Split

- **Train:** first 5 calls per ticker (~70 rows, roughly Q4-2023 – Q1-2025)
- **Test:** remaining calls (~61 rows, Q2-2025 onward)
- Split is by call index within each ticker, not randomly, to avoid look-ahead bias

---

## Known Caveats

- **Small test set:** n=46 after NaN filtering on guidance features. All metrics are directional.
- **PLTR:** uses a pre-recorded format with no Q&A section. Q&A-derived features are NaN for all PLTR calls.
- **Sentiment anchoring:** qwen3:8b assigns sentiment ≈ 0.85 to most calls. The `sentiment_delta` and `lm_sentiment` features partially compensate.
- **Guidance schema:** `cache/extractions/` contains the v2 structured guidance (per-line `{line, direction}` objects). An older v1 extraction (scalar `"guidance"` field) was produced during development but is not used by the model.
