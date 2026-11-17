"""Launch with `streamlit run app.py`."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from engine import (FIELDS, JACKPOT_DENOMINATOR, Ticket, validate_history,
                    uniform_ticket, score_components, score_ticket, top_scored_ticket,
                    monte_carlo, walk_forward_backtest, baseline_expected_matches)

st.set_page_config(page_title="EuroMillions Probability Lab", page_icon="🎲",layout="wide")
st.title("🎲 EuroMillions Probability Lab")
st.caption("Research and simulation tool — not a lottery predictor. Any valid 5+2 ticket has the same jackpot odds.")
st.warning("Never interpret a high selection-quality score as a higher chance of winning. Do not spend more based on these results.")

with st.sidebar:
    st.header("Configuration")
    seed = st.number_input("Random seed (reproducibility)",0,2**31-1,42,1)
    candidates = st.slider("Candidates scored per prediction",10,1000,100,10)
    st.subheader("Selection quality weights")
    weights = {key:st.slider(key.title(),0.,1.,default,.05) for key,default in
               [("spread",.25),("parity",.20),("coverage",.20),("spacing",.25),("frequency",.10)]}
    st.caption("Weights control preference, not lottery probabilities.")

st.header("1 · Import draw history")
st.write("Upload a CSV containing one draw per date and columns `date,n1,n2,n3,n4,n5,s1,s2`. Draws must use the current 5/50 + 2/12 format. The bundled CSV is **one user-provided example draw, not a historical dataset**.")
file = st.file_uploader("Historical draws CSV",type=["csv"])
try:
    raw = pd.read_csv(file) if file else pd.read_csv(Path(__file__).parent / "example_draw.csv")
    history = validate_history(raw)
    st.success(f"Validated {len(history):,} draw(s). Date range: {history.date.min().date()} to {history.date.max().date()}")
    with st.expander("Preview validated draw history"):
        st.dataframe(history,hide_index=True,use_container_width=True)
except (ValueError, pd.errors.ParserError, KeyError) as exc:
    st.error(f"CSV could not be used: {exc}")
    st.stop()

st.header("2 · Generate a candidate")
col1,col2=st.columns(2)
with col1:
    if st.button("Generate scored combination",type="primary"):
        rng=np.random.default_rng(int(seed))
        ticket, quality = top_scored_ticket(rng,weights,history,candidates)
        st.session_state["scored"]=(ticket,quality)
    if "scored" in st.session_state:
        ticket,quality=st.session_state["scored"]
        st.subheader(ticket.display())
        st.metric("Selection quality, not win probability",f"{quality:.1%}")
        st.bar_chart(pd.Series(score_components(ticket,history),name="component score"))
with col2:
    if st.button("Generate uniform random baseline"):
        st.session_state["random"] = uniform_ticket(np.random.default_rng(int(seed)+1))
    if "random" in st.session_state:
        st.subheader(st.session_state["random"].display())
        st.metric("Jackpot probability per valid line",f"1 / {JACKPOT_DENOMINATOR:,}")

st.header("3 · Monte Carlo simulator")
trials=st.select_slider("Independent simulated draws",options=[1_000,10_000,100_000,1_000_000],value=100_000)
if st.button("Run simulation"):
    simulated=monte_carlo(trials,int(seed))
    st.session_state["simulated"]=simulated
if "simulated" in st.session_state:
    simulated=st.session_state["simulated"]
    st.write("Match-count simulation uses exact independent hypergeometric sampling. It has the same distribution for any fixed valid ticket and a uniform random ticket.")
    pivot=simulated.pivot(index="main_matches",columns="star_matches",values="proportion")
    st.dataframe(pivot.style.format("{:.5%}"),use_container_width=True)
    st.bar_chart(simulated.groupby("main_matches")["proportion"].sum())
    st.caption("A jackpot is extremely rare; observing zero jackpots in even one million trials is normal.")

st.header("4 · Chronological walk-forward backtesting")
warmup=st.number_input("Earlier draws used before first test",min_value=1,max_value=5000,value=30)
if len(history)<=warmup:
    st.info(f"Upload at least {warmup+1} draw rows to evaluate held-out predictions. Current sample has {len(history)} row(s).")
else:
    st.caption("For each held-out draw, compare one scored ticket with one independent uniform-random ticket. Only earlier draws inform frequency statistics.")
    if st.button("Run walk-forward backtest"):
        st.session_state["backtest"]=walk_forward_backtest(history,weights,int(warmup),candidates,int(seed))
    if "backtest" in st.session_state:
        results=st.session_state["backtest"]
        summary=results.groupby("strategy").agg(draws=("total_matches","size"),
                            average_main_matches=("main_matches","mean"),
                            average_star_matches=("star_matches","mean"),
                            average_total_matches=("total_matches","mean"))
        st.dataframe(summary,use_container_width=True)
        st.line_chart(results.pivot(index="date",columns="strategy",values="total_matches").rolling(20,min_periods=1).mean())
        m,s=baseline_expected_matches()
        st.caption(f"Theoretical expected matches per random ticket: main {m:.3f}, stars {s:.3f}. Small differences are expected by chance. Tune parameters on one period and validate once on a fresh untouched period.")
        st.download_button("Download backtest results",results.to_csv(index=False),"backtest_results.csv","text/csv")

st.divider()
st.caption("Independent lottery draws do not remember previous outcomes. Structural scoring is a ticket-selection preference, not a prediction signal.")
