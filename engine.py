"""EuroMillions research engine. No predictive advantage is asserted."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable
import numpy as np
import pandas as pd

MAIN_POOL = 50
STAR_POOL = 12
JACKPOT_DENOMINATOR = 139_838_160
FIELDS = ["date", "n1", "n2", "n3", "n4", "n5", "s1", "s2"]


@dataclass(frozen=True)
class Ticket:
    mains: tuple[int, ...]
    stars: tuple[int, ...]

    def display(self) -> str:
        return f"{' - '.join(map(str,self.mains))} | Stars: {' - '.join(map(str,self.stars))}"


def validate_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate file and return chronological canonical draw table.

    Historical periods using different draw rules should not be mixed.
    """
    missing = [x for x in FIELDS if x not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
    out = df[FIELDS].copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid date; use YYYY-MM-DD.")
    out["date"] = dates
    for col in FIELDS[1:]:
        values = pd.to_numeric(out[col], errors="coerce")
        if values.isna().any() or not np.all(np.equal(values, np.floor(values))):
            raise ValueError(f"{col} has a missing or non-integer value")
        out[col] = values.astype(int)
    for _, row in out.iterrows():
        mains = [int(row[f"n{i}"]) for i in range(1,6)]
        stars = [int(row[f"s{i}"]) for i in range(1,3)]
        if len(set(mains)) != 5 or not all(1 <= n <= 50 for n in mains):
            raise ValueError(f"Invalid five main numbers on {row['date'].date()}")
        if len(set(stars)) != 2 or not all(1 <= s <= 12 for s in stars):
            raise ValueError(f"Invalid two Lucky Stars on {row['date'].date()}")
    if out["date"].duplicated().any():
        raise ValueError("Duplicate draw dates in CSV")
    return out.sort_values("date").reset_index(drop=True)


def uniform_ticket(rng: np.random.Generator) -> Ticket:
    mains = tuple(sorted(int(x) for x in rng.choice(np.arange(1,51), size=5, replace=False)))
    stars = tuple(sorted(int(x) for x in rng.choice(np.arange(1,13), size=2, replace=False)))
    return Ticket(mains, stars)


def draw_ticket(row: pd.Series) -> Ticket:
    return Ticket(tuple(sorted(int(row[f"n{i}"]) for i in range(1,6))),
                  tuple(sorted(int(row[f"s{i}"]) for i in range(1,3))))


def match_counts(ticket: Ticket, result: Ticket) -> tuple[int,int]:
    return len(set(ticket.mains) & set(result.mains)), len(set(ticket.stars) & set(result.stars))


def score_components(ticket: Ticket, history: pd.DataFrame | None = None) -> dict[str,float]:
    """Ticket aesthetics only, NOT predicted winning probabilities."""
    a = np.asarray(ticket.mains, dtype=float)
    spread = float((a[-1] - a[0]) / 49)
    evens = sum(n % 2 == 0 for n in ticket.mains)
    parity = max(0., 1. - abs(evens - 2.5) / 2.5)
    bands = len(set((n-1)//10 for n in ticket.mains))
    coverage = bands / 5
    gaps = np.diff(a)
    # Penalize near-neighbor clumping but do not assert this helps draw odds.
    spacing = float(np.mean(np.minimum(gaps/8., 1.)))
    if history is not None and len(history):
        counts = np.zeros(51, dtype=float)
        for i in range(1,6):
            counts += np.bincount(history[f"n{i}"].to_numpy(), minlength=51)[:51]
        # Laplace smoothed observed frequencies, rescaled to [0,1]
        selected = np.mean([(counts[n]+1)/(5*len(history)+50) for n in ticket.mains])
        history_score = float(np.clip(selected * 10, 0, 1))
    else:
        history_score = .5
    return {"spread":spread, "parity":parity, "coverage":coverage,
            "spacing":spacing, "frequency":history_score}


def score_ticket(ticket: Ticket, weights: dict[str,float], history: pd.DataFrame | None = None) -> float:
    comps = score_components(ticket,history)
    total = sum(max(0., weights.get(k,0.)) for k in comps)
    if total == 0:
        return 0.
    return sum(max(0., weights.get(k,0.))*v for k,v in comps.items())/total


def top_scored_ticket(rng:np.random.Generator, weights:dict[str,float], history:pd.DataFrame | None,
                      candidate_count:int=100) -> tuple[Ticket,float]:
    if not 1 <= candidate_count <= 10000:
        raise ValueError("candidate_count must be between 1 and 10000")
    best = None
    best_score = -1.
    for _ in range(candidate_count):
        candidate = uniform_ticket(rng)
        val = score_ticket(candidate,weights,history)
        if val > best_score:
            best, best_score = candidate,val
    assert best is not None
    return best, best_score


def monte_carlo(trials:int,seed:int=42) -> pd.DataFrame:
    """Simulate match-count distribution using exact hypergeometric draws.
    This produces the same match-count law as uniformly sampling full tickets.
    """
    if not 1 <= trials <= 2_000_000:
        raise ValueError("trials must be between 1 and 2,000,000")
    rng = np.random.default_rng(seed)
    main_matches = rng.hypergeometric(5,45,5,size=trials)
    star_matches = rng.hypergeometric(2,10,2,size=trials)
    counts = np.zeros((6,3),dtype=int)
    np.add.at(counts,(main_matches,star_matches),1)
    return pd.DataFrame([{"main_matches":m,"star_matches":s,"count":int(counts[m,s]),
                          "proportion":float(counts[m,s]/trials)}
                         for m in range(6) for s in range(3)])


def walk_forward_backtest(history:pd.DataFrame, weights:dict[str,float], warmup:int=30,
                          candidates:int=100, seed:int=42) -> pd.DataFrame:
    """Exactly one algorithm ticket and one independent uniform ticket per held-out draw.
    The scoring engine only sees earlier rows; changing weights after reviewing results
    still invites researcher overfitting, so confirm using a separate test period.
    """
    if warmup < 1: raise ValueError("warmup must be at least 1")
    if len(history) <= warmup: raise ValueError(f"Need > {warmup} historical draws for backtesting")
    rng = np.random.default_rng(seed)
    results = []
    for idx in range(warmup,len(history)):
        prior = history.iloc[:idx]
        actual = draw_ticket(history.iloc[idx])
        scored,quality = top_scored_ticket(rng,weights,prior,candidates)
        baseline = uniform_ticket(rng)
        for label,ticket in (("Scored",scored),("Uniform random",baseline)):
            m,s = match_counts(ticket,actual)
            results.append({"date":history.iloc[idx]["date"],"strategy":label,
                            "main_matches":m,"star_matches":s,"total_matches":m+s,
                            "score":quality if label=="Scored" else np.nan,
                            "ticket":ticket.display()})
    return pd.DataFrame(results)


def baseline_expected_matches() -> tuple[float,float]:
    return 5*5/50, 2*2/12
