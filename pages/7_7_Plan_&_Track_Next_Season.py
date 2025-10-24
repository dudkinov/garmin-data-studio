import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from core.utils import (
    build_calendar, 
    season_start_year,
    season_dates
)
from core.data import (
    load_data, 
    load_sport_data, 
    load_season_zone_mix, 
    load_weekly_time_in_zones
) 
from core.utils import show_reconnect_button
from core.tables import (
    smoothed_week_vector, 
    compute_sport_distribution, 
    prepare_weekly_zone_pivot, 
    compute_zone_distribution_comparison
)
from core.visuals import (
    plot_plan_vs_real, 
    plot_cumulative_progress, 
    plot_microcycle_bars, 
    plot_execution_heatmap, 
    plot_sport_pie, 
    plot_zone_distribution_bars, 
    plot_weekly_hr_zones
)
from core.constants import (
    THIS_SEASON, 
    BASELINE_SEASON, 
    UPLIFT, 
    SMOOTH_WINDOW, 
    TOL_BAND,
    SPORT_COLORS, ZONE_ORDER, ZONE_COLORS
)

# --- DB reconnect button
show_reconnect_button()

# --- Page 7 Config --- 
st.set_page_config(layout="wide", page_title="Planning & Tracking 2025/26", page_icon="📈")
st.title("📈 Plan & Track — Season 2025/26")
st.caption("Plan built from prior seasons + microcycle logic: progressive overload with every 4th week deload. First deload happens on week 2. Season total = last season × 1.10.")

# --- Load and prepare data --- 
# Load overall weekly totals
df_raw = load_data()

# Build base training calendar and merge with raw weekly data
calendar_df = build_calendar(df_raw)
df = (
    calendar_df
    .merge(df_raw, on=["training_season", "week_start"], how="left")
    .fillna(0)
    .sort_values(["training_season", "week_start"])
)

# Compute sequential week index per season
df["week_index"] = df.groupby("training_season").cumcount() + 1
df["week_start"] = pd.to_datetime(df["week_start"]).dt.tz_localize(None)

# Build ordered list of all seasons and find previous one
all_seasons = sorted(df["training_season"].dropna().unique(), key=season_start_year)
past_seasons = [s for s in all_seasons if s != THIS_SEASON]
last_season = past_seasons[-1] if past_seasons else None

# Pivot weekly training time across seasons (for comparisons / baselines)
pivot_time = (
    df.pivot_table(
        index="week_index",
        columns="training_season",
        values="training_time",
        aggfunc="sum"
    )
    .sort_index()
    .fillna(0.0)
)

# Fixed month ticks (Sep→Aug) for visualization reference
season_start = pd.Timestamp(year=2020, month=9, day=1)
fixed_months = pd.date_range(season_start, periods=12, freq="MS")
month_ticks = [
    int(((m - season_start) / np.timedelta64(1, "W"))) + 1 for m in fixed_months
]
month_labels = [m.strftime("%b") for m in fixed_months]

# --- Generate plan --- 
with st.expander("🧠 Planning method", expanded=False):
    st.markdown("""
- **Macro shape:** average 10-week smoothed weekly hours from the last two complete seasons (if available).
- **Microcycles:** 4-week pattern anchored so week 2 is a deload, then every 4 weeks (2, 6, 10, …).
  Mapping inside each 4-week block:
  - deload → 0.70  
  - next weeks → 0.96, 1.00, 1.06  
- **Season total:** scaled so sum(plan) = 2024/25 × 1.10.  
- **Tolerance band:** ±10 % around the planned line for tracking.
    """)

# --- Build macro shape from last 2 seasons ---
shape_sources = past_seasons[-2:] if len(past_seasons) >= 2 else past_seasons
if shape_sources:
    stacks = np.stack([smoothed_week_vector(pivot_time, s, SMOOTH_WINDOW) for s in shape_sources], axis=0)
    base_shape = stacks.mean(axis=0)
else:
    base_shape = np.ones(52)

base_shape = np.clip(base_shape, 0, None)
if base_shape.sum() <= 0:
    base_shape = np.ones(52)
base_shape /= base_shape.sum()

# --- Apply 4-week microcycle pattern (week 2 deload anchor) ---
cycle_map = {0: 0.70, 1: 0.96, 2: 1.00, 3: 1.06}
cycle_mult = np.array([cycle_map[(w - 2) % 4] for w in range(1, 53)])
micro_shape = (base_shape * cycle_mult) / (base_shape * cycle_mult).sum()

# --- Scale to baseline total × uplift ---
baseline_total = float(pivot_time.get(BASELINE_SEASON, pd.Series()).sum() or 0.0)
target_total = baseline_total * UPLIFT
plan_series = pd.Series(micro_shape * target_total, index=np.arange(1, 53), name="plan_hours")

# --- Build tolerance bands and actual data ---
band_low = plan_series * (1 - TOL_BAND)
band_high = plan_series * (1 + TOL_BAND)
real_series = pivot_time.get(THIS_SEASON, pd.Series()).reindex(range(1, 53), fill_value=0.0)

# --- KPI Cards ---
today = date.today()
date_str = today.strftime("%b %d, %Y")

# Week number (in this season)
df_2526 = df[df["training_season"] == THIS_SEASON]
# --- Compute current week dynamically based on season start
season_start = pd.Timestamp(year=2025, month=9, day=1)
cur_week = int(((pd.Timestamp(today) - season_start).days // 7) + 1)

# Guard if we have data only up to previous week
if not df_2526.empty:
    max_week_in_data = df_2526["week_index"].max()
    cur_week = max(cur_week, int(max_week_in_data))

# YTD totals & YoY vs baseline season (same week)
ytd_hours = float(pivot_time.get(THIS_SEASON, pd.Series()).loc[:cur_week].sum()) if cur_week else 0.0
ytd_last  = float(pivot_time.get(BASELINE_SEASON, pd.Series()).loc[:cur_week].sum()) if (cur_week and BASELINE_SEASON in pivot_time.columns) else np.nan
yoy_pct   = (ytd_hours / ytd_last * 100 - 100) if (np.isfinite(ytd_last) and ytd_last > 0) else np.nan
yoy_delta = f"{yoy_pct:+.1f}%" if np.isfinite(yoy_pct) else None

# Guard for plan-based values
_plan = locals().get("plan_series", None)
_real = locals().get("real_series", None)

# Cumulative on-track % (real vs plan up to current week)
on_track_pct = np.nan
if (cur_week and isinstance(_plan, pd.Series) and isinstance(_real, pd.Series)):
    plan_cum = float(_plan.loc[:cur_week].sum())
    real_cum = float(_real.loc[:cur_week].sum())
    on_track_pct = (real_cum / plan_cum * 100) if plan_cum > 0 else np.nan

# This-week target/real and attainment %
this_week_target = float(_plan.loc[cur_week]) if (cur_week and isinstance(_plan, pd.Series)) else np.nan
this_week_real   = float(_real.loc[cur_week]) if (cur_week and isinstance(_real, pd.Series)) else np.nan
tw_attain_pct    = ((this_week_real / this_week_target) * 100) if (np.isfinite(this_week_target) and this_week_target > 0) else np.nan
tw_delta         = (f"{(tw_attain_pct-100):+.1f}%" if np.isfinite(tw_attain_pct) else None)

c1, c2, c3, c4 = st.columns(4)

# 1) Week card
with c1:
    st.metric("🗓️ Current week", str(cur_week) if cur_week else "—")
    st.markdown(f"**:blue[{date_str}]**")

# 2) YTD hours card (+ progress to plan if available)
with c2:
    st.metric("⏱️ YTD hours", f"{ytd_hours:,.0f} h")
    if np.isfinite(on_track_pct):
        st.progress(min(max(on_track_pct/100, 0), 1), text=f"On-track: {on_track_pct:.0f}%")

# 3) YoY vs baseline (same week)
with c3:
    st.metric(f"📊 vs {BASELINE_SEASON} (same week)", yoy_delta if yoy_delta else "—", delta=yoy_delta, delta_color="normal")
    if np.isfinite(ytd_last):
        st.caption(f"Last year: {ytd_last:,.0f} h")

# 4) This week — target vs real (+ attainment)
with c4:
    val = f"{this_week_target:.1f}h / {this_week_real:.1f}h" if np.isfinite(this_week_target) and np.isfinite(this_week_real) else "—"
    st.metric("🎯 This week — target / real", val, delta=tw_delta, delta_color="normal")
    if np.isfinite(tw_attain_pct):
        st.progress(min(max(tw_attain_pct/100, 0), 1), text=f"Attainment: {tw_attain_pct:.0f}%")

# this week target / real
this_week_target = float(plan_series.loc[cur_week]) if cur_week else np.nan
this_week_real = float(real_series.loc[cur_week]) if cur_week else np.nan
on_track = np.nan
if cur_week and plan_series.loc[:cur_week].sum() > 0:
    on_track = float(real_series.loc[:cur_week].sum() / plan_series.loc[:cur_week].sum() * 100)

st.divider()

# Row 1 — Main plan vs real + cumulative
st.subheader("📅 Weekly Plan vs Real")
r1c1, r1c2 = st.columns([1.35, 1])

# --- Weekly Plan vs Real (chart)
with r1c1:
    plot_plan_vs_real(plan_series, band_low, band_high, real_series, month_ticks, month_labels)

# --- Cumulative chart
with r1c2:
    plan_cum = plan_series.cumsum()
    real_cum = real_series.cumsum()
    plot_cumulative_progress(plan_cum, real_cum, month_ticks, month_labels, cur_week, on_track)

st.markdown("---")

# --- Row 2 — Microcycle Structure ---
st.subheader("⚙️ Microcycle Structure & Execution")
r2c1, r2c2 = st.columns([1.1, 1.25])

# Microcycle bars
from core.visuals import plot_microcycle_bars
with r2c1:
    plot_microcycle_bars(plan_series, month_ticks, month_labels)

# Execution heatmap (Real / Plan %)
from core.visuals import plot_execution_heatmap
with r2c2:
    plot_execution_heatmap(plan_series, real_series)

st.markdown("---")

# --- Row 3 — Sport mix (planned vs real YTD) ---
st.subheader("⚖️ Sport Distribution — Planned vs Real YTD")

from core.visuals import plot_sport_pie
from core.tables import compute_sport_distribution

# --- Compute planned and real distributions
plan_dist, real_dist, comp = compute_sport_distribution(load_sport_data(), past_seasons, THIS_SEASON)

# --- Layout with two pies + comparison table
cA, cB, cC = st.columns([1, 1, 1.6])
with cA:
    st.pyplot(plot_sport_pie(plan_dist, "Planned mix (from prior seasons)"), use_container_width=False)
with cB:
    st.pyplot(plot_sport_pie(real_dist, "Real YTD"), use_container_width=False)
with cC:
    st.dataframe(comp.round(1), use_container_width=True)

st.markdown("---")
st.caption("Plan = macro trend (10-wk smoothed) × microcycle multipliers; first deload on week 2, then every 4 weeks. Total = 2024/25 × 1.10.")

# --- Chart: Intensity — Time in HR Zones
st.divider()
st.header("🔥 Intensity — Time in HR Zones")

# --- Weekly stacked area for THIS_SEASON ---
s_start, s_end = season_dates(THIS_SEASON)
df_week = load_weekly_time_in_zones(s_start, s_end)

# --- Align to calendar weeks of THIS_SEASON ---
cal_weeks = (
    calendar_df[calendar_df["training_season"] == THIS_SEASON][["week_start", "training_season"]]
      .copy()
)
cal_weeks["week_start"] = pd.to_datetime(cal_weeks["week_start"]).dt.date

if df_week.empty:
    st.info("No heart-rate zone data for this season yet.")
else:
    pv = prepare_weekly_zone_pivot(df_week, cal_weeks, zone_order=ZONE_ORDER)

    st.subheader(f"Weekly Time in HR Zones — {THIS_SEASON}")
    fig_tz = plot_weekly_hr_zones(pv, ZONE_ORDER, ZONE_COLORS, month_ticks, month_labels)
    st.pyplot(fig_tz, use_container_width=True)

# --- YTD Zone Distribution — Real vs Planned
st.subheader("YTD Zone Distribution — Real vs Planned")

# --- Compute plan vs real comparison ---
comp = compute_zone_distribution_comparison(
    load_season_zone_mix,
    past_seasons=past_seasons,
    this_season=THIS_SEASON,
    zone_order=ZONE_ORDER
)

# --- Layout: bar chart + table ---
c1, c2 = st.columns([1.1, 1.2])

with c1:
    fig_bar = plot_zone_distribution_bars(comp, ZONE_ORDER)
    st.pyplot(fig_bar, use_container_width=True)

with c2:
    st.markdown("#### 📋 Zone Mix Table")
    st.dataframe(
        comp.rename(columns={"plan_pct": "Planned %", "real_pct": "Real %"})
            .assign(**{
                "Planned %":        lambda d: d["Planned %"].round(1),
                "Real %":           lambda d: d["Real %"].round(1),
                "Δ (real - plan)":  lambda d: d["Δ (real - plan)"].round(1),
            }),
        use_container_width=True
    )
