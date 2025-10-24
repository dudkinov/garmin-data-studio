import pandas as pd
import numpy as np
import streamlit as st
from core.data import load_indexes_from_db, load_sport_data
from core.insights import generate_meta_insights
from core.visuals import draw_radar_chart
from core.tables import show_meta_table
from core.utils import show_reconnect_button

# DB reconnect button
show_reconnect_button()

EXCLUDED_SEASONS = {"2025/26"}

def normalize_to_100(series):
    max_val = series.max(skipna=True)
    if pd.isna(max_val) or max_val <= 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series / max_val * 100).round(1)

def safe_mean(values):
    vals = [v for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def build_meta_dataframe():
    """Build meta dataframe with Consistency, Periodisation, Volume, Intensity, Balance."""
    df_weeks = load_sport_data()
    df_idx_raw = load_indexes_from_db()

    # --- Seasonal Volume ---
    season_hours = (
        df_weeks.groupby("training_season", dropna=True)["training_time"]
        .sum()
        .reset_index(name="season_hours")
    )
    season_hours = season_hours[~season_hours["training_season"].isin(EXCLUDED_SEASONS)]
    season_hours["Volume"] = normalize_to_100(season_hours["season_hours"])

    # --- Indexes ---
    df_idx = df_idx_raw.copy()
    df_idx = df_idx[(df_idx["scope"] == "overall") & (~df_idx["training_season"].isin(EXCLUDED_SEASONS))]
    df_idx["calculated_on"] = pd.to_datetime(df_idx["calculated_on"], errors="coerce")

    idx_wide = (
        df_idx.sort_values("calculated_on")
        .groupby(["training_season", "index_name"], as_index=False)
        .last()
        .pivot(index="training_season", columns="index_name", values="index_value")
        .apply(pd.to_numeric, errors="coerce")
    )

    CONSISTENCY_KEYS = ["annual_consistency_index"]
    PERIODISATION_KEYS = ["sipi", "season_volume_periodisation_index", "ipi_intensity"]
    BALANCE_KEYS = ["balance_score"]
    INTENSITY_COL = "intensity_balance"

    seasons = idx_wide.index.astype(str)
    meta = pd.DataFrame(index=seasons)

    meta["Consistency"] = [safe_mean([idx_wide.loc[s, k] for k in CONSISTENCY_KEYS if k in idx_wide.columns]) for s in seasons]
    meta["Periodisation"] = [safe_mean([idx_wide.loc[s, k] for k in PERIODISATION_KEYS if k in idx_wide.columns]) for s in seasons]
    meta["Balance"] = [safe_mean([idx_wide.loc[s, k] for k in BALANCE_KEYS if k in idx_wide.columns]) for s in seasons]

    # --- Intensity ---
    if INTENSITY_COL in idx_wide.columns:
        meta["Intensity"] = idx_wide[INTENSITY_COL]
    else:
        meta["Intensity"] = np.nan  # simplified fallback

    # --- Merge Volume ---
    meta = meta.merge(season_hours.set_index("training_season")[["Volume"]], left_index=True, right_index=True, how="left")
    meta = meta[["Consistency", "Volume", "Intensity", "Periodisation", "Balance"]]
    meta = meta.applymap(lambda x: np.clip(x, 0, 100) if pd.notna(x) else np.nan)

    return meta, season_hours

# --- Page 6 Layout ---
st.set_page_config(layout="wide")
st.title("🧩 META-Dashboard")
st.caption("Aggregated performance metrics showing how each season evolved in structure, volume, and balance.")

# --- Load data ---
meta, season_hours = build_meta_dataframe()
all_seasons = meta.index.tolist()
latest, prev = all_seasons[-1], all_seasons[-2] if len(all_seasons) > 1 else None

# --- Scorecards ---
cols = st.columns(len(meta.columns))
for i, col in enumerate(meta.columns):
    now = meta.loc[latest, col]
    before = meta.loc[prev, col] if prev else None
    delta = (now - before) if before is not None else None
    emoji = ["📅","⏱️","🔥","📈","⚖️"][i]
    cols[i].metric(f"{emoji} {col}", f"{now:.1f}" if now else "—", f"{delta:+.1f}" if delta else "—")

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # --- Focus selector ---
    focus = st.radio("", ["All"] + all_seasons, horizontal=True, label_visibility="collapsed")
    show_meta_table(meta, focus)

with col1:
    draw_radar_chart(meta, all_seasons, focus)

with col3:
    generate_meta_insights(meta, season_hours)

