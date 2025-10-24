import streamlit as st
import numpy as np
import pandas as pd
from core.data import load_scope_data
from core.visuals import plot_seasonal_hours_sessions
from core.utils import safe_metric_value
from core.tables import build_summary_table
from core.utils import show_reconnect_button

# DB reconnect button
show_reconnect_button()

#Load data
df = load_scope_data()
if df.empty:
    st.error("No data available in the database.")
    st.stop()

# --- Data preparation for KPIs ---
default_weeks = 52 

# Ensure chronological order
df = df.sort_values("training_season", kind="stable").reset_index(drop=True)
first_season = df.loc[0, "training_season"]

# Ensure integer typed
df["weeks_in_season"] = df["weeks_in_season"].astype(int)

# Keep real weeks for the first season; hard-code others
df.loc[df["training_season"] != first_season, "weeks_in_season"] = default_weeks

# --- Recompute metrics that depend on weeks ---
for col in ["total_hours", "num_trainings", "weeks_in_season"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Derived metrics with safe division
df["avg_duration"] = (
    df["total_hours"] / df["num_trainings"]
).replace([np.inf, -np.inf], np.nan).round(2)

df["hours_per_week"] = (
    df["total_hours"] / df["weeks_in_season"]
).replace([np.inf, -np.inf], np.nan).round(2)

df["YoY_hours_%"] = df["total_hours"].pct_change() * 100
df["YoY_trainings_%"] = df["num_trainings"].pct_change() * 100

# --- KPI totals ---
tot_hours = df["total_hours"].sum()
tot_sessions = int(df["num_trainings"].sum()) if df["num_trainings"].sum() > 0 else 0

avg_week_all = (
    df["total_hours"].sum() / df["weeks_in_season"].sum()
    if df["weeks_in_season"].sum() > 0 else np.nan
)

avg_sess_all = (
    df["total_hours"].sum() / df["num_trainings"].sum()
    if df["num_trainings"].sum() > 0 else np.nan
)

#--- Page 1 layout --- 
st.set_page_config(layout="wide")
st.title("📊 Analysis Scope")
st.caption("Defines total training time, season-to-season progression, and links to detailed analysis pages.")

col1, col2, col3 = st.columns([1.2, 2, 1.1], gap="large")

# Left col: KPI panel
with col1:
    st.subheader("🏁 Scope KPIs")
    a, b = st.columns(2)
    a.metric("⏱ Total Hours", f"{tot_hours:,.0f} h")
    b.metric("🏋️ Sessions", f"{tot_sessions:,}")

    c, d = st.columns(2)
    c.metric("📆 Avg / Week", safe_metric_value(avg_week_all, "{:.1f} h"))
    d.metric("⚙️ Avg / Session", safe_metric_value(avg_sess_all, "{:.2f} h"))

    # 3rd row: YoY KPIs (vs previous season)
    e, f = st.columns(2)

    if len(df) >= 2:
        latest_hours      = df["total_hours"].iloc[-1]
        latest_sessions   = int(df["num_trainings"].iloc[-1])
        yoy_hours         = df["YoY_hours_%"].iloc[-1]
        yoy_sessions      = df["YoY_trainings_%"].iloc[-1]

        e.metric("📈 YoY Hours", f"{latest_hours:,.0f} h", delta=f"{yoy_hours:+.1f}%")
        f.metric("📊 YoY Sessions", f"{latest_sessions:,}", delta=f"{yoy_sessions:+.1f}%")
    else:
        e.metric("📈 YoY Hours", "—", delta="—")
        f.metric("📊 YoY Sessions", "—", delta="—")

# Center col: chart
with col2:
    st.subheader("📈 Season Totals — Hours & Trainings")
    fig = plot_seasonal_hours_sessions(df)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)


# Right col: links
with col3:
    st.subheader("🧭 Links to Analysis")
    st.page_link("pages/2_2_Volume.py",       label="Volume & Periodisation", icon="📊")
    st.page_link("pages/3_3_Consistency.py",  label="Consistency Dashboard",  icon="📈")
    st.page_link("pages/4_4_Intensity.py",    label="Intensity & HR Zones",   icon="🔥")
    st.page_link("pages/5_5_Sport_Balance.py",label="Sport Balance",          icon="🏊‍♂️")
    st.page_link("pages/6_6_Meta_Pannel.py",  label="META Dashboard",         icon="🧩")

# Summary table
styled = build_summary_table(df)

st.divider()
st.dataframe(styled, use_container_width=True)

