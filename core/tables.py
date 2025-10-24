import pandas as pd
import numpy as np
import streamlit as st
from pandas.io.formats.style import Styler
from core.utils import style_consistent, season_sort_key
from core.constants import ZONE_ORDER

# Page 1: value colors for summary table
def highlight_yoy(val):
    """Color positive YoY green, negative red."""
    if pd.isna(val):
        return ""
    color = "#00C853" if val > 0 else "#FF5252"
    return f"color: {color}; font-weight: bold;"

# Page 1: summary table
def build_summary_table(df: pd.DataFrame) -> Styler:
    summary = (
        df.set_index("training_season")[
            [
                "weeks_in_season", "num_trainings", "total_hours",
                "hours_per_week", "avg_duration", "YoY_hours_%", "YoY_trainings_%"
            ]
        ]
        .rename(columns={
            "weeks_in_season": "Weeks",
            "num_trainings": "Sessions",
            "total_hours": "Hours",
            "hours_per_week": "Avg h/week",
            "avg_duration": "Avg h/session"
        })
        .rename_axis("Season")
    )

    styled = (
        summary.style
        .format({
            "Weeks": "{:.0f}",
            "Sessions": "{:,.0f}",
            "Hours": "{:,.1f}",
            "Avg h/week": "{:.2f}",
            "Avg h/session": "{:.2f}",
            "YoY_hours_%": "{:+.1f}%",
            "YoY_trainings_%": "{:+.1f}%"
        })
        .applymap(highlight_yoy, subset=["YoY_hours_%", "YoY_trainings_%"])
        .background_gradient(subset=["Avg h/week"], cmap="YlGnBu")
    )

    return styled

# Consistency page: section 1
def compute_training_frequency_index(df_consistency):
    """Compute training frequency index (TF) per training season."""
    if df_consistency.empty:
        return pd.DataFrame(columns=["training_season", "TF"])

    freq_df = df_consistency.groupby("training_season").apply(
        lambda x: 100 * (1 - x.loc[x["activities"] == 0, "days_count"].sum() / x["days_count"].sum())
    ).reset_index(name="TF")
    freq_df["TF"] = freq_df["TF"].round(1)
    return freq_df

# Consistency page: section 2
def compute_streak_summary(df_streaks):
    """Aggregate streak statistics by training season."""
    if df_streaks.empty:
        return pd.DataFrame(columns=["training_season", "avg_streak", "max_streak", "n_streaks"])

    return (
        df_streaks.groupby("training_season")
        .agg(
            avg_streak=("streak_length", "mean"),
            max_streak=("streak_length", "max"),
            n_streaks=("streak_length", "count")
        )
        .reset_index()
    )

# Consistency page: section 2
def compute_streak_penalty_index(df_streaks):
    """Compute Streak Penalty (SP) index per training season."""
    if df_streaks.empty:
        return pd.DataFrame(columns=["training_season", "SP"])

    streak_summary = (
        df_streaks.groupby("training_season")
        .agg(avg_streak=("streak_length", "mean"), max_streak=("streak_length", "max"))
        .reset_index()
    )
    streak_summary["SP"] = 100 - (5 * streak_summary["avg_streak"] + 1.5 * streak_summary["max_streak"])
    streak_summary["SP"] = streak_summary["SP"].clip(0, 100)
    return streak_summary

# Consistency page: section 3
def compute_long_session_summary(df_long_summary):
    """Compute total long sessions and average duration per season."""
    if df_long_summary.empty:
        return pd.DataFrame(columns=["training_season", "total_long", "avg_dur"])

    return (
        df_long_summary.groupby("training_season")
        .agg(
            total_long=("num_long_activities", "sum"),
            avg_dur=("avg_duration", "mean"),
        )
        .reset_index()
    )

def compute_long_session_balance(df_long_summary):
    """Compute Long-Session Balance (LSB) index per training season."""
    if df_long_summary.empty:
        return pd.DataFrame(columns=["training_season", "LSB"])

    lsb = (
        df_long_summary.groupby("training_season")["num_long_activities"]
        .sum()
        .reset_index(name="long_sessions")
    )
    max_ls = lsb["long_sessions"].max() if len(lsb) else 1
    lsb["LSB"] = (lsb["long_sessions"] / max_ls * 100).round(1)
    return lsb

# Consistency page: Section 4
def compute_long_timeline(df_long_timeline):
    """Prepare continuous daily timeline with color mapping."""
    if df_long_timeline.empty:
        return pd.DataFrame()

    df = df_long_timeline.copy()
    df["date"] = df["start_time"].dt.floor("D")

    calendar = pd.DataFrame({
        "date": pd.date_range(df["date"].min(), df["date"].max(), freq="D", tz="UTC")
    })
    merged = pd.merge(calendar, df, on="date", how="left")

    color_map = {
        "cycling": "tab:blue",
        "running": "tab:orange",
        "swimming": "tab:green",
        "none": "lightgray",
    }
    merged["sport"] = merged["sport"].fillna("none")
    merged["avg_duration"] = merged["avg_duration"].fillna(0)
    merged["color"] = merged["sport"].map(color_map)
    return merged

def compute_distribution_consistency_index(df_long_timeline, get_training_season):
    """Compute Distribution Consistency Index (DCI) per training season."""
    if df_long_timeline.empty:
        return pd.DataFrame(columns=["training_season", "active_weeks", "total_weeks", "DCI"])

    df = df_long_timeline.copy()
    df["date"] = df["start_time"].dt.floor("D")
    df["training_season"] = df["date"].apply(get_training_season)
    df["week"] = df["date"].dt.isocalendar().week

    weekly_dist = (
        df.groupby("training_season")["week"]
        .nunique()
        .reset_index(name="active_weeks")
    )

    total_weeks = (
        df.groupby("training_season")["date"]
        .apply(lambda x: max(1, (x.max() - x.min()).days / 7))
        .reset_index(name="total_weeks")
    )

    dist_summary = weekly_dist.merge(total_weeks, on="training_season", how="left")
    dist_summary["DCI"] = (dist_summary["active_weeks"] / dist_summary["total_weeks"] * 100).clip(0, 100).round(1)
    return dist_summary


# Final consistency index calc
def compute_sub_indexes(df_consistency, df_streaks, df_long_summary, df_long_timeline, get_training_season):
    """Compute all sub-indexes: TF, SP, LS, DCI."""
    # TF — Training Frequency
    tf = (
        df_consistency.groupby("training_season")
        .apply(lambda x: 100 * (1 - x.loc[x["activities"] == 0, "days_count"].sum() / x["days_count"].sum()))
        .reset_index(name="TF")
    )

    # SP — Streak Penalty
    sp = (
        df_streaks.groupby("training_season")
        .agg(avg_streak=("streak_length", "mean"), max_streak=("streak_length", "max"))
        .reset_index()
    )
    sp["SP"] = 100 - (5 * sp["avg_streak"] + 1.5 * sp["max_streak"])
    sp["SP"] = sp["SP"].clip(0, 100)

    # LS — Long Session Volume
    ls = (
        df_long_summary.groupby("training_season")["num_long_activities"]
        .sum()
        .reset_index(name="long_sessions")
    )
    max_ls = ls["long_sessions"].max() if len(ls) else 1
    ls["LS"] = (ls["long_sessions"] / max_ls * 100).round(1)

    # DCI — Distribution Consistency
    df_tl = df_long_timeline.copy()
    df_tl["date"] = df_tl["start_time"].dt.floor("D")
    df_tl["training_season"] = df_tl["date"].apply(get_training_season)
    df_tl["week"] = df_tl["date"].dt.isocalendar().week
    weekly = df_tl.groupby("training_season")["week"].nunique().reset_index(name="active_weeks")
    total = df_tl.groupby("training_season")["date"].apply(
        lambda x: max(1, (x.max() - x.min()).days / 7)
    ).reset_index(name="total_weeks")
    dci = weekly.merge(total, on="training_season")
    dci["DCI"] = (dci["active_weeks"] / dci["total_weeks"] * 100).clip(0, 100).round(1)

    return tf, sp, ls, dci


def combine_consistency_indexes(tf, sp, ls, dci, dci2):
    """Merge all sub-indexes + DCI2 (if available) and compute final Consistency Index."""
    import pandas as pd

    # Handle missing or malformed df_dci2 safely ---
    if dci2 is None or dci2.empty:
        dci2 = pd.DataFrame(columns=["training_season", "DCI2"])
    else:
        # Normalize column names to lowercase
        dci2.columns = [c.lower().strip() for c in dci2.columns]

        # Try to detect correct training season column
        if "training_season" not in dci2.columns:
            # Sometimes comes back as 'trainingseason' or 'season'
            possible_cols = [c for c in dci2.columns if "season" in c]
            if possible_cols:
                dci2 = dci2.rename(columns={possible_cols[0]: "training_season"})
            else:
                dci2["training_season"] = None  # placeholder

        # Try to detect value column (DCI2)
        if "dci2" not in dci2.columns:
            value_col = next((c for c in dci2.columns if c in ["index_value", "dci", "deload_consistency_index_2"]), None)
            if value_col:
                dci2 = dci2.rename(columns={value_col: "DCI2"})
            else:
                dci2["DCI2"] = 0.0  # fallback if missing

        # Keep only required columns
        dci2 = dci2[["training_season", "DCI2"]]

    # --- Merge everything ---
    combined = (
        tf.merge(sp[["training_season", "SP"]], on="training_season", how="outer")
          .merge(ls[["training_season", "LS"]], on="training_season", how="outer")
          .merge(dci[["training_season", "DCI"]], on="training_season", how="outer")
          .merge(dci2, on="training_season", how="left")
          .fillna(0)
    )

    # Ensure numeric and clean
    for col in ["TF", "SP", "LS", "DCI", "DCI2"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    # --- Weighted formula ---
    combined["Consistency_Index"] = (
        0.35 * combined["TF"]
        + 0.25 * combined["SP"]
        + 0.20 * combined["LS"]
        + 0.10 * combined["DCI"]
        + 0.10 * combined["DCI2"]
    ).round(1)

    return combined

# Intensity page: HR zone distribution
def prepare_hr_zone_distribution(df, zone_order):
    """Prepare cumulative and per-season HR zone distributions."""

    # ✅ ensure numeric dtype even if cloud environment reads strings
    df["total_time_in_zone"] = pd.to_numeric(df["total_time_in_zone"], errors="coerce")

    # --- Cumulative
    agg_cum = (
        df.groupby("hr_zone")["total_time_in_zone"]
        .sum()
        .reindex(zone_order, fill_value=0)
        .reset_index()
    )
    agg_cum["total_time_in_zone"] = agg_cum["total_time_in_zone"].astype(float)
    total_cum = agg_cum["total_time_in_zone"].sum()
    agg_cum["pct_time_in_zone"] = np.where(
        total_cum > 0,
        (agg_cum["total_time_in_zone"] / total_cum) * 100,
        0
    ).round(1)

    # --- Seasonal
    agg_season = (
        df.groupby(["training_season", "hr_zone"])["total_time_in_zone"]
        .sum()
        .reset_index()
    )
    if not agg_season.empty:
        # ✅ convert again just in case groupby restored object dtype
        agg_season["total_time_in_zone"] = pd.to_numeric(
            agg_season["total_time_in_zone"], errors="coerce"
        )
        agg_season["pct_time_in_zone"] = (
            agg_season.groupby("training_season")["total_time_in_zone"]
            .transform(lambda x: (x / x.sum()) * 100)
        ).round(1)
    else:
        agg_season["pct_time_in_zone"] = 0.0

    pivot_season = (
        agg_season
        .pivot(index="training_season", columns="hr_zone", values="pct_time_in_zone")
        .fillna(0)
        .reindex(columns=zone_order, fill_value=0)
    )

    return agg_cum, pivot_season


# Summary table, meta-page
def show_meta_table(meta_df: pd.DataFrame, focus: str = "All"):
    """
    Display META dashboard table with unified 40–100 color scale,
    rounded to 1 decimal, and fade-out effect for non-focused seasons.
    """
    def fade(col: pd.Series) -> list[str]:
        if focus in ("All", col.name):
            return [''] * len(col)
        return ['color: #999; background-color: #f8f8f8;'] * len(col)

    styled = (
        style_consistent(meta_df.T)
        .format("{:.1f}")       
        .apply(fade, axis=0)    
    )

    st.dataframe(styled, use_container_width=True)

# Season consistency index table
def show_index_table(df: pd.DataFrame, value_col: str = "Consistency_Index") -> Styler:
    """
    Return a styled table using the global 40–100 color scale.
    """
    num_cols = df.select_dtypes(include=["number"]).columns
    fmt = {c: "{:.1f}" for c in num_cols}

    styled = df.style.format(fmt)
    styled = style_consistent(styled, subset=[value_col])
    return styled

# Intensity page: section 2
def prepare_weekly_intensity(df_view: pd.DataFrame) -> pd.DataFrame:
    """Prepare weekly HR-zone intensity distribution as percentages per week."""
    if df_view.empty:
        return pd.DataFrame()

    df = df_view.copy()
    df["intensity_group"] = df["hr_zone"].map({
        "Zone 1": "Low", "Zone 2": "Low",
        "Zone 3": "Moderate", "Zone 4": "Moderate",
        "Zone 5": "High"
    })

    trend = (
        df.groupby(["week_start", "intensity_group"])["total_time_in_zone"]
        .sum()
        .reset_index()
    )
    trend["pct"] = trend.groupby("week_start")["total_time_in_zone"].transform(
        lambda x: (x / x.sum()) * 100
    )

    pivot = (
        trend.pivot(index="week_start", columns="intensity_group", values="pct")
        .fillna(0)
        .sort_index()
    )

    return pivot

# Intensity page: section 3
def prepare_seasonal_intensity_balance(df_base: pd.DataFrame, selected_seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-sport and per-season intensity balance vs 80/15/5 target."""
    if df_base.empty:
        return pd.DataFrame(), pd.DataFrame()

    core_sports = ["cycling", "running", "swimming"]
    df = df_base[df_base["sport"].isin(core_sports)].copy()
    if selected_seasons:
        df = df[df["training_season"].isin(selected_seasons)]

    df["intensity_group"] = df["hr_zone"].map({
        "Zone 1": "Low", "Zone 2": "Low",
        "Zone 3": "Moderate", "Zone 4": "Moderate",
        "Zone 5": "High"
    })

    agg = (
        df.groupby(["sport", "training_season", "intensity_group"])["total_time_in_zone"]
        .sum().reset_index()
    )
    agg["pct"] = agg.groupby(["sport", "training_season"])["total_time_in_zone"].transform(
        lambda x: (x / x.sum()) * 100
    )

    pivot_bal = (
        agg.pivot_table(index=["training_season", "sport"], columns="intensity_group", values="pct")
        .fillna(0)[["Low", "Moderate", "High"]]
    )

    # Score vs 80/15/5 target
    ideal = {"Low": 80, "Moderate": 15, "High": 5}
    weights = {"Low": 0.4, "Moderate": 0.3, "High": 0.3}

    def total_score(row):
        score = 0
        for z in ["Low", "Moderate", "High"]:
            diff = abs(row.get(z, 0) - ideal[z])
            comp = max(0, 100 - (diff / ideal[z]) * 100)
            score += comp * weights[z]
        return round(min(score, 100), 1)

    pivot_bal["Score"] = pivot_bal.apply(total_score, axis=1)
    overall_scores = (
        pivot_bal.groupby("training_season")["Score"]
        .mean().reset_index().rename(columns={"Score": "Triathlon_Score"})
    )

    return pivot_bal, overall_scores


def show_seasonal_balance_tables(pivot_bal: pd.DataFrame, overall_scores: pd.DataFrame):
    """Display main and overall balance tables with unified style."""
    if pivot_bal.empty:
        st.info("No balance data available for selected filters.")
        return

    styled_main = (
        style_consistent(pivot_bal.reset_index(), subset=["Score"])
        .format({"Low": "{:.1f}%", "Moderate": "{:.1f}%", "High": "{:.1f}%", "Score": "{:.1f}"})
    )
    st.dataframe(styled_main, use_container_width=True)

    st.markdown("### 🏁 Overall Seasonal Intensity Balance Index")
    styled_overall = (
        style_consistent(overall_scores.set_index("training_season"), subset=["Triathlon_Score"])
        .format({"Triathlon_Score": "{:.1f}"})
    )
    st.dataframe(styled_overall, use_container_width=True)

# Intensity page: Section 4
def compute_intensity_periodisation_index(df_base: pd.DataFrame, selected_seasons: list[str]) -> pd.DataFrame:
    """Compute the Intensity Periodisation Index (IPI) based on quarter-to-quarter growth."""
    if df_base.empty:
        return pd.DataFrame()

    core_sports = ["cycling", "running", "swimming"]
    dfc = df_base[df_base["sport"].isin(core_sports)].copy()
    if selected_seasons:
        dfc = dfc[dfc["training_season"].isin(selected_seasons)]

    zone_weights = {"Zone 1": 1, "Zone 2": 2, "Zone 3": 3, "Zone 4": 4, "Zone 5": 5}

    wk = (
        dfc.groupby(["training_season", "week_start", "hr_zone"])["total_time_in_zone"]
        .sum().reset_index()
    )

    def fill_missing_zones(g):
        g = g.set_index("hr_zone").reindex(zone_weights.keys(), fill_value=0).reset_index()
        g["training_season"] = g["training_season"].iloc[0]
        g["week_start"] = g["week_start"].iloc[0]
        return g

    wk = wk.groupby(["training_season", "week_start"], group_keys=False).apply(fill_missing_zones).reset_index(drop=True)

    wk["total_week"] = wk.groupby(["training_season", "week_start"])["total_time_in_zone"].transform("sum")
    wk["pct"] = np.where(wk["total_week"] > 0, wk["total_time_in_zone"] / wk["total_week"] * 100, 0)
    wk["weighted"] = wk["pct"] * wk["hr_zone"].map(zone_weights)
    wk["weighted"] = wk["weighted"].astype(float)
    weekly = wk.groupby(["training_season", "week_start"])["weighted"].sum().reset_index()
    weekly["weighted"] = weekly["weighted"].astype(float)
    weekly["intensity_score"] = weekly["weighted"] / 100.0


    def assign_quarters(g):
        g = g.sort_values("week_start").reset_index(drop=True)
        n = len(g)
        g["quarter"] = np.floor(np.arange(n) * 4 / max(n, 1)).astype(int) + 1
        g["quarter"] = g["quarter"].clip(1, 4)
        return g

    weekly = weekly.groupby("training_season", group_keys=False).apply(assign_quarters)
    q = weekly.groupby(["training_season", "quarter"])["intensity_score"].mean().unstack().fillna(0)
    q.columns = [f"Q{i}_Intensity" for i in q.columns]
    q = q.reset_index()

    # --- Calculate quarter ratios and IPI scoring
    ratios = []
    for i, row in q.iterrows():
        prev_q4 = q.iloc[i - 1]["Q4_Intensity"] if i > 0 else None
        next_q1 = q.iloc[i + 1]["Q1_Intensity"] if i < len(q) - 1 else None

        def safe_ratio(a, b):
            if a is None or b is None or not np.isfinite(a) or b <= 0:
                return None
            return round(a / b, 3)

        r_prev = safe_ratio(row["Q1_Intensity"], prev_q4)
        r_12 = safe_ratio(row["Q2_Intensity"], row["Q1_Intensity"])
        r_23 = safe_ratio(row["Q3_Intensity"], row["Q2_Intensity"])
        r_34 = safe_ratio(row["Q4_Intensity"], row["Q3_Intensity"])
        r_next = safe_ratio(next_q1, row["Q4_Intensity"])

        seq = [r_prev, r_12, r_23, r_34, r_next]
        positives = [r for r in seq if r and r > 1.0]
        total_pos = len(positives)
        max_run, cur = 0, 0
        for r in seq:
            if r and r > 1.0:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0

        if max_run >= 3: ipi = 100
        elif max_run == 2: ipi = 75
        elif total_pos >= 3: ipi = 70
        elif total_pos == 2: ipi = 60
        elif total_pos == 1: ipi = 50
        else: ipi = 0

        ratios.append({
            "training_season": row["training_season"],
            "R_prev(Q1/Q4_prev)": r_prev,
            "R_12(Q2/Q1)": r_12,
            "R_23(Q3/Q2)": r_23,
            "R_34(Q4/Q3)": r_34,
            "R_next(Q1_next/Q4)": r_next,
            "Positives": total_pos,
            "Max_consecutive": max_run,
            "Intensity_Periodisation_Index": ipi
        })

    ipi_final = pd.DataFrame(ratios).sort_values("training_season", key=lambda s: [season_sort_key(x) for x in s])
    return ipi_final

def show_intensity_periodisation_table(ipi_final: pd.DataFrame):
    """Display IPI results with unified color formatting."""
    if ipi_final.empty:
        st.info("No quarterly data available to score IPI.")
        return

    styled = (
        style_consistent(ipi_final, subset=["Intensity_Periodisation_Index"], vmin=0, vmax=100)
        .format({
            "R_prev(Q1/Q4_prev)": "{:.3f}",
            "R_12(Q2/Q1)": "{:.3f}",
            "R_23(Q3/Q2)": "{:.3f}",
            "R_34(Q4/Q3)": "{:.3f}",
            "R_next(Q1_next/Q4)": "{:.3f}",
        })
    )

    st.dataframe(styled, use_container_width=True)

    st.caption("""
    **Interpretation**
    - 📈 Ratios > 1.0 → rising quarter-to-quarter intensity (Z3–Z5 share)
    - 🔄 Consecutive positives indicate structured periodisation
    - 🟢 100 = consistent build · 🟡 75–70 = partial · 🔴 <50 = flat/declining
    """)

# Plan and track page 7: smoothed weekly vector for a given season
def smoothed_week_vector(pivot_time, season: str, smooth_window: int = 10) -> np.ndarray:
    """Return smoothed weekly vector of training time for a given season."""
    series = pivot_time.get(season, pd.Series()).reindex(range(1, 53), fill_value=0.0)
    return series.rolling(window=smooth_window, center=True, min_periods=1).mean().values

# # Plan and track page 7: sport distribution
def compute_sport_distribution(df_sport: pd.DataFrame, past_seasons: list, this_season: str):
    """Compute planned and real sport mix distributions and comparison table."""
    last_two = past_seasons[-2:] if len(past_seasons) >= 2 else past_seasons

    # Planned mix = mean of last two complete seasons
    if last_two:
        sd = (
            df_sport[df_sport["training_season"].isin(last_two)]
            .groupby(["training_season", "sport"])["training_time"].sum()
            .reset_index()
        )
        sd["percent"] = sd.groupby("training_season")["training_time"].transform(lambda x: x / x.sum() * 100)
        plan_dist = sd.groupby("sport")["percent"].mean().reset_index()
    else:
        plan_dist = pd.DataFrame({
            "sport": ["cycling", "running", "swimming", "training"],
            "percent": [40, 30, 20, 10]
        })

    # Real YTD
    real_dist = (
        df_sport[df_sport["training_season"] == this_season]
        .groupby("sport")["training_time"].sum()
        .reset_index()
    )
    total = real_dist["training_time"].sum()
    real_dist["percent"] = (
        real_dist["training_time"] / total * 100 if total > 0 else 0
    )

    # Comparison
    comp = pd.merge(plan_dist, real_dist, on="sport", how="outer", suffixes=("_plan", "_real")).fillna(0)
    comp["Δ (real - plan)"] = comp["percent_real"] - comp["percent_plan"]

    return plan_dist, real_dist, comp

# Page 7: zone distribution chart
def prepare_weekly_zone_pivot(df_week: pd.DataFrame, cal_weeks: pd.DataFrame, zone_order: list[str]) -> pd.DataFrame:
    """Merge HR zone weekly data with calendar weeks and create week index pivot."""
    pv = (
        df_week
        .pivot_table(index="week_start", columns="hr_zone", values="hours", aggfunc="sum")
        .reindex(columns=zone_order)
        .fillna(0.0)
        .reset_index()
    )

    wk_map = cal_weeks.copy()
    wk_map["week_index"] = range(1, len(wk_map) + 1)
    pv = wk_map.merge(pv, on="week_start", how="left").fillna(0.0)
    return pv.sort_values("week_index")

# Page 7: zone distribution comparison
def compute_zone_distribution_comparison(load_func, past_seasons, this_season, zone_order):
    """Compute planned vs real HR zone mix comparison for YTD."""
    # --- Real for current season ---
    real_mix = (
        load_func(this_season)[["hr_zone", "percent"]]
        .rename(columns={"percent": "real_pct"})
    )

    # --- Planned = average of last two complete seasons ---
    last_two = past_seasons[-2:] if len(past_seasons) >= 2 else past_seasons
    if last_two:
        mixes = [
            load_func(ss)[["hr_zone", "percent"]].rename(columns={"percent": ss})
            for ss in last_two
        ]
        plan_df = mixes[0]
        for m in mixes[1:]:
            plan_df = plan_df.merge(m, on="hr_zone", how="outer")
        plan_df = plan_df.fillna(0)
        plan_df["plan_pct"] = plan_df.drop(columns=["hr_zone"]).mean(axis=1)
        plan_mix = plan_df[["hr_zone", "plan_pct"]]
    else:
        # fallback if no historical data
        plan_mix = pd.DataFrame({
            "hr_zone": zone_order,
            "plan_pct": [60, 25, 10, 4, 1]
        })

    # --- Combine plan and real ---
    comp = (
        pd.DataFrame({"hr_zone": zone_order})
        .merge(plan_mix, on="hr_zone", how="left")
        .merge(real_mix, on="hr_zone", how="left")
        .fillna(0.0)
    )

    # --- Delta ---
    comp["Δ (real - plan)"] = comp["real_pct"] - comp["plan_pct"]

    return comp

