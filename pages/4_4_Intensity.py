import streamlit as st
import pandas as pd
from core.data import (
    get_engine, 
    load_hr_zone_data_weekly,
    build_index_upload_rows, 
    upsert_training_indexes
)
from core.visuals import (
    plot_zone_distribution, 
    plot_seasonal_zone_distribution,
    plot_weekly_intensity
)
from core.utils import (
    get_training_season, 
    season_sort_key
)
from core.insights import (
    generate_hr_zone_insights,
    generate_weekly_intensity_insights,
    generate_seasonal_balance_insights,
    generate_intensity_periodisation_insights
)
from core.tables import (
    prepare_hr_zone_distribution, 
    prepare_weekly_intensity,
    prepare_seasonal_intensity_balance,
    show_seasonal_balance_tables,
    show_intensity_periodisation_table,
    compute_intensity_periodisation_index
)
from core.constants import ZONE_ORDER, ZONE_COLORS
from core.utils import show_reconnect_button

# DB reconnect button
show_reconnect_button()

# --- Load data
df_base = load_hr_zone_data_weekly()
df_base["week_start"] = pd.to_datetime(df_base["week_start"]).dt.tz_localize(None)
df_base["training_season"] = df_base["week_start"].apply(get_training_season)
    
# --- Page 4: Intensity Config
st.set_page_config(layout="wide")
st.title("📊 Intensity Analysis")
st.caption("Analysis focuses on intensity distribution and periodisation around each season. It's based on the time spent in Heart Rate Zones")

# --- Filters
col_filters = st.columns([2, 3])

with col_filters[0]:
    sports = ["Overall"] + sorted(df_base["sport"].unique())
    selected_sport = st.radio("Select sport:", sports, index=0, horizontal=True)

with col_filters[1]:
    all_seasons = sorted(df_base["training_season"].dropna().unique().tolist(), key=season_sort_key)
    selected_seasons = st.multiselect("Select seasons:", all_seasons, default=all_seasons)

st.divider()

# Build a working view for charts (respect filters)
df_view = df_base.copy()
if selected_sport != "Overall":
    df_view = df_view[df_view["sport"] == selected_sport]
if selected_seasons:
    df_view = df_view[df_view["training_season"].isin(selected_seasons)]

# --- Section 1: HR Zone Distribution (Cumulative + Seasonal)
st.subheader("📘 HR Zone Distribution — Cumulative & Seasonal")

# --- 1. Prepare data
agg_cum, pivot_season = prepare_hr_zone_distribution(df_view, ZONE_ORDER)

# --- 2. Visuals
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("**Cumulative HR Zone Proportions**")
    fig_cum = plot_zone_distribution(agg_cum, "Cumulative", highlight=True)
    st.pyplot(fig_cum, use_container_width=True)

with col_right:
    st.markdown("**Per-Season Comparison**")
    fig_seasonal = plot_seasonal_zone_distribution(pivot_season, selected_sport, ZONE_COLORS)
    st.pyplot(fig_seasonal, use_container_width=True)

# --- 3. Insights
with st.expander("🔎 Insights — Zone Distribution", expanded=True):
    insights = generate_hr_zone_insights(agg_cum, pivot_season, season_sort_key)
    for line in insights:
        st.markdown(line)

st.divider()

# --- Section 2: Weekly Intensity Periodization (stacked area + insights)
st.subheader("📉 Weekly Intensity Periodization (Low / Moderate / High)")

# Prepare data and draw chart
pivot_trend = prepare_weekly_intensity(df_view)
fig_trend = plot_weekly_intensity(pivot_trend)
st.pyplot(fig_trend, use_container_width=True)

# Insights block
with st.expander("🔎 Insights — Weekly Periodisation", expanded=True):
    insights = generate_weekly_intensity_insights(pivot_trend)
    for line in insights:
        st.markdown(line)

st.divider()

# --- Section 3: Seasonal Intensity Balance vs 80/15/5
st.subheader("⚖️ Seasonal Intensity Balance vs. 80/15/5 (Triathlon core sports)")

pivot_bal, overall_scores = prepare_seasonal_intensity_balance(df_base, selected_seasons)
show_seasonal_balance_tables(pivot_bal, overall_scores)

with st.expander("🔎 Insights — Seasonal Balance vs 80/15/5", expanded=True):
    insights = generate_seasonal_balance_insights(pivot_bal, overall_scores)
    for line in insights:
        st.markdown(line)

st.divider()

# Intensity Periodisation Index (IPI)
st.subheader("🔥 Intensity Periodisation Index (IPI_Intensity) — Quarter-to-Quarter Growth")

ipi_final = compute_intensity_periodisation_index(df_base, all_seasons)
ipi_final = ipi_final.fillna(0)
show_intensity_periodisation_table(ipi_final)

with st.expander("🔎 Insights — Intensity Periodisation Index", expanded=True):
    insights = generate_intensity_periodisation_insights(ipi_final)
    for line in insights:
        st.markdown(line)

st.divider()

# --- Save indexes to DB ---
try:
    engine = get_engine()
    upload_rows = build_index_upload_rows(overall_scores, ipi_final)
    msg = upsert_training_indexes(engine, upload_rows)
    st.success(msg)
except Exception as e:
    st.error(f"❌ Failed to save indexes to DB: {e}")
