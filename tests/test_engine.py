import numpy as np
import pandas as pd
import pytest
from engine import (uniform_ticket,validate_history, score_ticket,
                    top_scored_ticket,monte_carlo,walk_forward_backtest,
                    baseline_expected_matches)

WEIGHTS={"spread":.25,"parity":.2,"coverage":.2,"spacing":.25,"frequency":.1}


def sample_history(n=40):
    rng=np.random.default_rng(99)
    rows=[]
    for date in pd.date_range('2026-01-01',periods=n,freq='D'):
        t=uniform_ticket(rng)
        rows.append([str(date.date()),*t.mains,*t.stars])
    return validate_history(pd.DataFrame(rows,columns=["date","n1","n2","n3","n4","n5","s1","s2"]))


def test_random_validity_and_seed():
    r=np.random.default_rng(5)
    tickets=[uniform_ticket(r) for _ in range(500)]
    assert all(len(set(t.mains))==5 and len(set(t.stars))==2 for t in tickets)
    assert all(1<=min(t.mains) and max(t.mains)<=50 for t in tickets)
    assert all(1<=min(t.stars) and max(t.stars)<=12 for t in tickets)
    assert uniform_ticket(np.random.default_rng(5))==tickets[0]


def test_invalid_history():
    df=sample_history(2)
    df.loc[0,"n2"]=df.loc[0,"n1"]
    with pytest.raises(ValueError): validate_history(df)


def test_scoring_and_monte_carlo():
    history=sample_history()
    t,quality=top_scored_ticket(np.random.default_rng(3),WEIGHTS,history,50)
    assert 0<=quality<=1
    assert abs(quality-score_ticket(t,WEIGHTS,history))<1e-12
    out=monte_carlo(3000,6)
    assert out['count'].sum()==3000
    assert abs(out['proportion'].sum()-1)<1e-10


def test_backtest_is_walk_forward():
    h=sample_history()
    out=walk_forward_backtest(h,WEIGHTS,warmup=30,candidates=15)
    assert len(out)==20
    assert out.groupby('date').size().eq(2).all()
    assert tuple(round(x,4) for x in baseline_expected_matches())==(.5,.3333)
