import pandas as pd
import numpy as np
import streamlit as st
import time
from pandas.io.formats.style import Styler
from core.constants import COLOR_MAP, COLOR_MIN, COLOR_MAX

# Reconnect DB button
def show_reconnect_button():
    """
    Displays a sidebar button that clears cached resources
    and reconnects to the database when pressed.
    """
    if st.sidebar.button("🔄 Reconnect Database"):
        st.cache_resource.clear()      
        st.cache_data.clear()           
        st.success("✅ Cache cleared. Reconnecting to database...")
        time.sleep(1)
        st.rerun()         

# Full calendar to merge with df
def build_calendar(df):
    calendars = []
    for season, row in df.groupby("training_season")["week_start"].agg(["min","max"]).iterrows():
        if pd.isna(row["min"]) or pd.isna(row["max"]):
            continue
        weeks = pd.date_range(start=row["min"], end=row["max"], freq="W-MON")
        calendars.append(pd.DataFrame({"week_start": weeks, "training_season": season}))
    calendar_df = pd.concat(calendars, ignore_index=True)
    return calendar_df

# Page 1: ensures correct data type for KPIs when none 
def safe_metric_value(value, fmt="{:.2f} h"):
        """Return formatted metric string or '—' for invalid values."""
        try:
            val = float(value)
            return fmt.format(val) if np.isfinite(val) else "—"
        except (TypeError, ValueError):
            return "—"
        
# Page 2: compute week index relative to fixed season start (Sept 1)
def get_season_start(season_label: str) -> pd.Timestamp:
    start_year = int(season_label.split("/")[0])
    return pd.Timestamp(year=start_year, month=9, day=1)

# Splits overall timelime to custom training seasons
def get_training_season(date):
    """Return training season string like 2024/25 based on month."""
    return f"{date.year}/{str((date.year + 1) % 100).zfill(2)}" if date.month >= 9 else f"{date.year - 1}/{str(date.year % 100).zfill(2)}"

# Page 2: helpers
def safe_ratio(num, den):
    if den in (None, 0) or not np.isfinite(den): return np.nan
    if num is None or not np.isfinite(num): return np.nan
    return num / den

def longest_true_run(bools):
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best

# Consistency page, helper for insights
def _latest_prev(seasons):
    """Return (latest, prev) for a chronologically sorted season list like ['2020/21','2021/22',...]"""
    if not seasons: return None, None
    srt = sorted(seasons)
    latest = srt[-1]
    prev = srt[-2] if len(srt) >= 2 else None
    return latest, prev

# Page 4: color mapping for HR chart
zone_colors = {
    "Zone 1": "grey",
    "Zone 2": "blue",
    "Zone 3": "green",
    "Zone 4": "orange",
    "Zone 5": "red"
}

# Consistensy page: styling for plots
def apply_matplotlib_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "regular",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "legend.frameon": True,
        "legend.edgecolor": "lightgray",
        "axes.prop_cycle": plt.cycler("color", [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
        ])
    })

# Intensity page: helper
def season_sort_key(s: str) -> int:
    try:
        return int(s.split("/")[0])
    except Exception:
        return 0
    
# Ensures all your tables have the same color interpretation
def style_consistent(obj, subset=None, cmap=None, vmin=None, vmax=None) -> Styler:
    """
    Apply the unified 40–100 color gradient to either a DataFrame or a Styler.
    - If `obj` is a DataFrame, convert to Styler.
    - If `obj` is a Styler, operate on it directly.
    """
    # ensure we have a Styler
    styler: Styler = obj if isinstance(obj, Styler) else obj.style

    # default subset = all columns (if DataFrame was passed without subset)
    if subset is None and isinstance(obj, pd.DataFrame):
        subset = obj.columns

    return styler.background_gradient(
        subset=subset,
        cmap=cmap or COLOR_MAP,
        vmin=COLOR_MIN if vmin is None else vmin,
        vmax=COLOR_MAX if vmax is None else vmax,
    )

# Helper: extract starting year from training season string (e.g. "2024/25" → 2024)
def season_start_year(s: str) -> int:
    return int(s.split("/")[0]) if isinstance(s, str) and "/" in s else 0

# Page 7: Helper for time in zone chart 
def season_dates(season_label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return start and end timestamps for a given training season label (e.g. '2024/25')."""
    y = int(season_label.split("/")[0])
    start = pd.Timestamp(year=y, month=9, day=1)
    end = pd.Timestamp(year=y + 1, month=9, day=1)
    return start, end

