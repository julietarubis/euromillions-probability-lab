# EuroMillions Probability Lab

An educational Streamlit dashboard with four engines:

- **Scoring:** configurable candidate scoring (spread, parity, 10-number band coverage, spacing, and smoothed historical frequency). Scores represent preference, **not** estimated probabilities.
- **Uniform random baseline:** each legal five-main/two-star ticket is equally likely.
- **Monte Carlo:** samples match counts using independent hypergeometric distributions, identical in law to simulated independent lottery draws.
- **Chronological backtesting:** at each held-out date, compare one scored ticket and one uniform baseline ticket, with the scored strategy seeing **only earlier draws**.

## Run locally

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m pytest -q
```

## CSV specification

Columns: `date,n1,n2,n3,n4,n5,s1,s2`

One draw per date, ISO date, five **distinct** numbers from 1–50, two **distinct** Lucky Stars from 1–12. Other columns ignored. Duplicate dates are rejected. Dates are sorted ascending. The bundled `example_draw.csv` contains **just the single draw supplied in the conversation**, not an independently verified historical dataset. Upload a trustworthy historical CSV to run meaningful walk-forward backtesting. Ensure all uploaded draws use the same 5/50 + 2/12 format (different historical game formats must not be mixed).

## Interpretation and caveats

- A single line has a jackpot probability of exactly **1 / 139,838,160** under current 5/50 + 2/12 rules.
- Scoring can prefer *visually balanced* tickets but does not increase jackpot odds.
- The simulation uses analytical hypergeometric **match-count** sampling: it does not claim to generate future draw numbers.
- Backtesting compares **one** algorithm ticket against **one** baseline ticket per held-out draw. The results are very noisy and should not be interpreted as proof of an edge.
- Do not pick/tune scoring weights from reported test performance and then treat the same test as independent verification. Set weights in advance; evaluate a later untouched period.
- The provided draw is user-supplied and unverified; no fabricated historical results are included.
- Prize payouts and prize-tier rules are not modelled, and no profitability claims are made.
