import streamlit as st
import pandas as pd
import numpy as np
from core.data import (
    load_sport_data, 
    get_engine, 
    compute_deload_consistency, 
    save_indexes_to_db, 
    compute_svpi
)
from core.visuals import (
    plot_weekly_metric, 
    plot_trend_metric, 
    plot_deload_consistency, 
    plot_svpi_bar_chart
)
from core.utils import (
    build_calendar, 
    get_season_start
)
from core.insights import (
    generate_weekly_deload_insights, 
    generate_seasonal_trend_insights
)
from core.utils import show_reconnect_button

# DB reconnect button
show_reconnect_button()

# --- Load and prepare data
EXCLUDED_SEASONS = {"2025/26"}

# --- Load and calendarize ---
df_raw = load_sport_data()
calendar_df = build_calendar(df_raw)

# --- Filter out unwanted seasons early ---
df_raw = df_raw[~df_raw["training_season"].isin(EXCLUDED_SEASONS)]

# --- Ensure each sport has all season-week combinations ---
sports_df = df_raw[["sport"]].drop_duplicates()
calendar_full = calendar_df.merge(sports_df, how="cross")

df = (
    calendar_full
    .merge(
        df_raw[["training_season", "week_start", "sport", "training_time"]],
        on=["training_season", "week_start", "sport"],
        how="left"
    )
    .fillna({"training_time": 0.0})
)

# Compute week index within each season
def compute_week_index(row):
    if pd.isna(row["week_start"]):
        return np.nan
    delta = (row["week_start"] - get_season_start(row["training_season"])) / np.timedelta64(1, "W")
    return int(delta) + 1

df["week_index"] = df.apply(compute_week_index, axis=1)
df = df[df["week_index"] >= 1].copy()

# Chart ticks and lables
SEASON_START = pd.Timestamp(year=2020, month=9, day=1)
MONTH_COUNT = 12

fixed_months = pd.date_range(SEASON_START, periods=MONTH_COUNT, freq="MS")
month_ticks = [
    int(((m - SEASON_START) / np.timedelta64(1, "W"))) + 1 for m in fixed_months
]
month_labels = [m.strftime("%b") for m in fixed_months]

# --- Page 2: Volume --- 
st.set_page_config(layout="wide")
st.title("📊 Volume Analysis")
st.caption("Consistency and Periodisation Analysis based on weekly time volume data.")

# --- Filters
col_filters = st.columns([2, 3])

with col_filters[0]:
    sports = ["Overall"] + sorted(df["sport"].unique())
    selected_sport = st.radio("Select Sport:", sports, index=0, horizontal=True)

with col_filters[1]:
    all_seasons = sorted(df["training_season"].unique())
    default_seasons = [s for s in all_seasons if s not in EXCLUDED_SEASONS]

    selected_seasons = st.multiselect(
        "Select Seasons:",
        options=all_seasons,
        default=default_seasons
    )

    # Filter out excluded
    selected_seasons = [s for s in selected_seasons if s not in EXCLUDED_SEASONS]

    # Ensure at least one season is always selected
    if not selected_seasons:
        selected_seasons = default_seasons

# Filtered data and pivots
df_filtered = (
    df if selected_sport == "Overall"
    else df[df["sport"] == selected_sport]
).query("training_season in @selected_seasons")

pivot_time = (
    df_filtered
    .pivot_table(index="week_index", columns="training_season", values="training_time", aggfunc="sum")
    .sort_index()
)

# Smoothed version for charts (10-week rolling mean) 
pivot_time_smooth = pivot_time.rolling(window=10, min_periods=1).mean()

# Palette (only selected seasons)
color_map = {
    "2020/21": "#F6C90E",
    "2021/22": "#32CD32",
    "2022/23": "#FFA500",
    "2023/24": "#1E90FF",
    "2024/25": "#FF0000",
}

color_map = {s: c for s, c in color_map.items() if s in selected_seasons}

# --- Section 1: weekly time chart and deload consistency index
st.markdown("---")
st.markdown("## 🕓 Weekly Training Volume")
st.caption("Weekly volume in hours is used to determine consistency of deloads.")

col1, col2 = st.columns([2, 1])
with col1:
    fig_time_weekly = plot_weekly_metric(
        pivot_time[selected_seasons],
        ylabel="Training Hours",
        title="Weekly Training Time (Compared by Season)",
        colors=color_map,
        month_ticks=month_ticks,
        month_labels=month_labels,
        label_colors=["#FF0000"] * len(month_ticks),
    )
    st.pyplot(fig_time_weekly, use_container_width=True)
    st.caption(f"Filters applied: Sport = {selected_sport}, Seasons = {', '.join(selected_seasons)}")

with col2:
    # --- Prepare df_dload
    if selected_sport == "Overall":
        df_dload = (
            df.groupby(["training_season", "week_index"], as_index=False)
            .agg(training_time=("training_time", "sum"))
        )
    else:
        df_dload = (
            df[df["sport"] == selected_sport]
            .groupby(["training_season", "week_index"], as_index=False)
            .agg(training_time=("training_time", "sum"))
        )

    df_dload = df_dload[df_dload["training_season"] != "2025/26"]

    # --- Calculate deload metrics (moved to core.data)
    dci_df, deload_positions = compute_deload_consistency(df_dload)

    # --- Plot (moved to core.charts)
    fig_dci2 = plot_deload_consistency(dci_df)

    # --- Display
    with st.container():
        with st.expander("**🟡 Deload (Rest Week) Consistency Index — DCI₂**", expanded=True):
            st.caption("""
            Detects how regularly **deload weeks** (20–40% reduction in training time)  
            appear every ~4 weeks — showing planned recovery structure.
            """)
            with st.expander("📉 View Chart", expanded=True):
                st.pyplot(fig_dci2, use_container_width=True)

            with st.expander("📊 View Table", expanded=False):
                st.dataframe(
                    dci_df.style.background_gradient(subset=["DCI₂"], cmap="YlGn", vmin=50, vmax=100),
                    use_container_width=True,
                )

# ---Auto Insights: Weekly Volume + Deloads ----------
st.markdown("### 🔎 Insights — Weekly Volume & Deloads")

ins_weekly = generate_weekly_deload_insights(pivot_time, deload_positions, selected_seasons)
if ins_weekly:
    for line in ins_weekly:
        st.markdown(f"- {line}")
else:
    st.info("No weekly insights available for the current selection.")

# --- Section 2: Season Trend charts and periodization index
st.markdown("---")

st.markdown("## 🧭 Seasonal Time Trends")
st.caption("Trends are used to calculate Training Volume Periodisation Index.")
col1, col2 = st.columns([2, 1])

#Right col: Periodisation Index
with col2:
    svpi_df = compute_svpi(pivot_time, selected_seasons)
    fig_svpi = plot_svpi_bar_chart(svpi_df)

    with st.container():
        with st.expander("**🔁 Season Volume Periodisation Index (SVPI)**", expanded=True):
            st.caption("""
            Evaluates quarter-to-quarter growth in **weekly training time**,  
            showing how consistently seasonal volume builds and periodises.
            """)
            with st.expander("📉 View Chart", expanded=True):
                st.pyplot(fig_svpi, use_container_width=True)

            with st.expander("📊 View Table", expanded=False):
                st.dataframe(
                    svpi_df.style.background_gradient(subset=["SVPI"], cmap="RdYlGn", vmin=0, vmax=100),
                    use_container_width=True,
                )

#Left col: Trend chart
with col1:
    fig_time_trend = plot_trend_metric(
        pivot=pivot_time_smooth[selected_seasons],
        ylabel="Training Hours",
        title="Smoothed Training Time Trend (10-week rolling mean)",
        colors=color_map,
        month_ticks=month_ticks,
        month_labels=month_labels,
        label_colors=["#FF0000"] * len(month_ticks),
    )
    st.pyplot(fig_time_trend, use_container_width=True)
    st.caption(f"Filters applied: Sport = {selected_sport}, Seasons = {', '.join(selected_seasons)}")

# Insights Trends
st.markdown("### 🔎 Insights — Seasonal Trends")

ins_trend = generate_seasonal_trend_insights(svpi_df, pivot_time_smooth, selected_seasons)

if ins_trend:
    for line in ins_trend:
        st.markdown(f"- {line}")
else:
    st.info("No seasonal insights available for the current selection.")

# --- Save indexes to DB ---
st.markdown("---")

engine = get_engine()
upload_rows = []

# SVPI
for _, row in svpi_df.iterrows():
    upload_rows.append({
        "training_season": str(row["training_season"]),
        "index_name": "Season_Volume_Periodisation_Index",
        "scope": "overall",
        "index_value": float(row["SVPI"]),
        "source_page": "Volume_Analysis_Page",
    })

# DCI₂ (reused from core.data)
dci_df, _ = compute_deload_consistency(df)
for _, r in dci_df.iterrows():
    upload_rows.append({
        "training_season": str(r["training_season"]),
        "index_name": "Deload_Consistency_Index_2",
        "scope": "overall",
        "index_value": float(r["DCI₂"]) if np.isfinite(r["DCI₂"]) else 0.0,
        "source_page": "Volume_Analysis_Page",
    })

# Save
success, error = save_indexes_to_db(engine, upload_rows, "2_Volume_Page")

