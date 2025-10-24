import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from core.tables import style_consistent
from core.data import load_sport_data, get_engine, save_indexes_to_db
from core.utils import build_calendar, show_reconnect_button
from core.visuals import plot_stacked_distribution, plot_cumulative_pie, plot_balance_bar
from core.insights import generate_distribution_insights, generate_balance_insights
from core.constants import SPORT_COLORS

# DB reconnect button
show_reconnect_button()

# Page setup
st.set_page_config(layout="wide")
st.title("🧮 Sport Balance")
st.caption("Time distribution across triathlon disciplines and balance score vs. benchmark.")

# Load and prepare data
engine = get_engine()
df = load_sport_data()
calendar_df = build_calendar(df)
sports = df["sport"].drop_duplicates()
calendar_full = pd.merge(calendar_df, sports, how="cross")

df = pd.merge(calendar_full, df, on=["training_season", "week_start", "sport"], how="left")
df["training_time"] = df["training_time"].fillna(0)

sport_totals = (
    df.groupby(["training_season", "sport"])["training_time"]
    .sum()
    .reset_index()
)
sport_totals = sport_totals[sport_totals["training_season"] != "2025/26"]
sport_totals["percent"] = sport_totals.groupby("training_season")["training_time"].transform(
    lambda x: (x / x.sum() * 100) if x.sum() > 0 else 0
)

# Filters
all_seasons = sorted(sport_totals["training_season"].unique())
selected_seasons = st.multiselect("Select Seasons:", options=all_seasons, default=all_seasons)
if not selected_seasons:
    selected_seasons = all_seasons

pct_pivot = (
    sport_totals.pivot(index="training_season", columns="sport", values="percent")
    .reindex(all_seasons)
    .fillna(0)
)

# Section 1 — Distribution overview
st.markdown("---")
st.subheader("📊 Distribution Overview")
st.caption("Cumulative chart aggregates across selected seasons. Stacked bars show per-season distribution.")

col1, col2 = st.columns([1, 2.2])
with col1:
    st.pyplot(plot_cumulative_pie(sport_totals, selected_seasons), use_container_width=False)
with col2:
    st.pyplot(plot_stacked_distribution(sport_totals, selected_seasons), use_container_width=True)

st.markdown("#### 🔎 Insights — Distribution Overview")
insights = generate_distribution_insights(sport_totals, selected_seasons, pct_pivot)
if insights:
    for line in insights:
        st.markdown(f"- {line}")
else:
    st.info("No distribution insights available for the current selection.")

# Section 2 — Balance score calculation
st.markdown("---")
st.subheader("⚖️ Training Discipline Balance Score")
st.caption("Score vs benchmark: cycling 40%, running 30%, swimming 20%, training 10%.")

ideal = {"cycling": 40, "running": 30, "swimming": 20, "training": 10}
pivot_pct = sport_totals.pivot(index="training_season", columns="sport", values="percent").fillna(0).sort_index()

results = []
for season, row in pivot_pct.iterrows():
    deviation_sum = sum(abs(row.get(sp, 0) - val) for sp, val in ideal.items())
    deviation_penalty = deviation_sum / 2
    score = max(0, 100 - deviation_penalty)
    results.append({
        "training_season": season,
        **{sp: row.get(sp, 0) for sp in ideal.keys()},
        "Deviation (%)": round(deviation_penalty, 1),
        "Balance_Score": round(score, 1),
    })

balance_df = pd.DataFrame(results).sort_values("training_season", ascending=False)

# Section 3 — Display table and chart
col1, col2 = st.columns([2.3, 1])
with col1:
    st.dataframe(
        style_consistent(balance_df, subset=["Balance_Score"])
        .format({**{sp: "{:.1f}" for sp in ideal.keys()}, "Deviation (%)": "{:.1f}", "Balance_Score": "{:.1f}"}),
        use_container_width=True,
    )
with col2:
    plot_balance_bar(balance_df)

# Section 4 — Auto insights
st.markdown("#### 🔎 Insights — Balance Score")
balance_insights = generate_balance_insights(balance_df, all_seasons, pivot_pct, ideal)
if balance_insights:
    for line in balance_insights:
        st.markdown(f"- {line}")
else:
    st.info("No balance insights available for the current selection.")

# Section 5 — Save balance scores to DB
engine = get_engine()

upload_rows = []
for _, r in balance_df.iterrows():
    upload_rows.append({
        "training_season": str(r["training_season"]),
        "index_name": "Balance_Score",
        "scope": "overall",
        "index_value": float(r["Balance_Score"]),
    })

success, error = save_indexes_to_db(engine, upload_rows, "5_Sport_Balance")