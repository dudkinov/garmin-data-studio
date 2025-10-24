import numpy as np
import pandas as pd
import streamlit as st
from core.utils import season_sort_key

# Summary/Meta page
def generate_meta_insights(meta_df, season_hours):
    """Generate textual insights based on META metrics."""
    insights = []

    all_seasons = list(meta_df.index)
    if len(all_seasons) < 1:
        st.info("No insights available.")
        return

    latest = all_seasons[-1]
    prev = all_seasons[-2] if len(all_seasons) > 1 else None

    METRICS = ["Consistency", "Volume", "Intensity", "Periodisation", "Balance"]

    def val(season, metric):
        try:
            return float(meta_df.loc[season, metric])
        except Exception:
            return np.nan

    def mean_finite(values):
        vals = [v for v in values if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    # --- Current & previous season values ---
    latest_vals  = {m: val(latest, m) for m in METRICS}
    prev_vals    = {m: val(prev, m) for m in METRICS} if prev else {}
    overall_now  = mean_finite(latest_vals.values())
    overall_prev = mean_finite(prev_vals.values()) if prev else np.nan

    # --- Summary statements ---
    if np.isfinite(overall_now):
        if overall_now >= 85:
            insights.append(f"**{latest}** overall META is **excellent** ({overall_now:.1f}).")
        elif overall_now >= 70:
            insights.append(f"**{latest}** overall META is **solid** ({overall_now:.1f}).")
        else:
            insights.append(f"**{latest}** META shows **room for improvement** ({overall_now:.1f}).")

    # --- Strongest & weakest metrics ---
    finite_latest = {k: v for k, v in latest_vals.items() if np.isfinite(v)}
    if finite_latest:
        strongest = max(finite_latest, key=finite_latest.get)
        weakest = min(finite_latest, key=finite_latest.get)
        insights.append(
            f"Strongest: **{strongest}** ({finite_latest[strongest]:.1f}); "
            f"Weakest: **{weakest}** ({finite_latest[weakest]:.1f})."
        )

    # --- Year-over-year deltas ---
    if prev:
        deltas = {
            m: (latest_vals[m] - prev_vals[m])
            for m in METRICS
            if np.isfinite(latest_vals[m]) and np.isfinite(prev_vals[m])
        }
        if deltas:
            up = max(deltas, key=deltas.get)
            down = min(deltas, key=deltas.get)
            if deltas[up] > 0:
                insights.append(f"Biggest improvement vs **{prev}**: **{up}** ({deltas[up]:+.1f}).")
            if deltas[down] < 0:
                insights.append(f"Biggest drop vs **{prev}**: **{down}** ({deltas[down]:+.1f}).")
        if np.isfinite(overall_now) and np.isfinite(overall_prev):
            insights.append(f"Overall META change vs **{prev}**: {overall_now - overall_prev:+.1f}.")

    # --- Volume delta (raw hours) ---
    raw_now = season_hours.set_index("training_season").get("season_hours", {}).get(latest, np.nan)
    raw_prev = season_hours.set_index("training_season").get("season_hours", {}).get(prev, np.nan) if prev else np.nan
    if np.isfinite(raw_now) and np.isfinite(raw_prev):
        delta = raw_now - raw_prev
        trend = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        insights.append(f"Raw hours in **{latest}**: **{raw_now:.0f} h** ({trend} {delta:+.0f} h vs {prev}).")

    # --- Focus areas (<60) ---
    low = [m for m, v in latest_vals.items() if np.isfinite(v) and v < 60]
    if low:
        insights.append("Focus areas (<60): " + ", ".join(f"**{m}**" for m in low) + ".")

    # --- Output ---
    st.markdown("### 🔎 Insights")
    if insights:
        for line in insights:
            st.markdown(f"- {line}")
    else:
        st.info("No insights could be generated for the current selection.")


# Volume page insights (Weekly volume, deload index)  
def generate_weekly_deload_insights(pivot_time, deload_positions, selected_seasons):
    """Generate textual insights for Weekly Volume & Deload patterns."""
    insights, peaks, consistency, spikes = [], [], [], []

    for s in selected_seasons:
        ser = pivot_time[s].fillna(0.0)
        if ser.empty:
            continue

        peak_val = float(ser.max())
        peak_week = int(ser.idxmax()) if np.isfinite(ser.idxmax()) else None
        mean, std = ser.mean(), ser.std()
        cv = (std / mean * 100) if (np.isfinite(std) and mean > 0) else np.nan

        peaks.append((s, peak_val, peak_week))
        consistency.append((s, cv))

        sm = ser.rolling(10, min_periods=1).mean()
        spike_mask = ser > (sm * 1.4)
        spikes.append((s, int(spike_mask.sum())))

    if peaks:
        s_peak = max(peaks, key=lambda t: t[1])
        insights.append(f"Highest weekly peak: **{s_peak[0]}** — **{s_peak[1]:.1f} h** (week **{s_peak[2]}**).")

    if consistency:
        finite = [c for c in consistency if np.isfinite(c[1])]
        if finite:
            s_cons = min(finite, key=lambda t: t[1])
            insights.append(f"Most even weekly volume (lowest CV): **{s_cons[0]}** — **{s_cons[1]:.1f}%**.")

    if spikes:
        s_spike = max(spikes, key=lambda t: t[1])
        if s_spike[1] > 0:
            insights.append(f"Large weekly spikes (>40% above trend): **{s_spike[0]}** — **{s_spike[1]}** weeks (watch overreaching).")

    def mean_interval(positions):
        if len(positions) < 2:
            return np.nan
        return float(np.mean(np.diff(sorted(positions))))

    cadences = [(s, mean_interval(deload_positions.get(s, []))) for s in selected_seasons]
    finite_cad = [c for c in cadences if np.isfinite(c[1])]
    if finite_cad:
        best = min(finite_cad, key=lambda t: abs(t[1] - 4.0))
        insights.append(f"Deload cadence closest to **~4 weeks**: **{best[0]}** — avg **{best[1]:.1f} w** between deloads.")

    return insights

# Volume page, periodisation index insights: 
def generate_seasonal_trend_insights(svpi_df, pivot_time_smooth, selected_seasons):
    """Generate textual insights for SVPI and seasonal trends."""
    insights = []
    if not svpi_df.empty:
        best = svpi_df.loc[svpi_df["SVPI"].idxmax()]
        worst = svpi_df.loc[svpi_df["SVPI"].idxmin()]
        insights.append(f"Best periodisation: **{best['training_season']}** (SVPI **{best['SVPI']}**).")
        insights.append(f"Flattest build: **{worst['training_season']}** (SVPI **{worst['SVPI']}**).")

    if not selected_seasons:
        return insights

    def season_sort_key(s): return int(s.split("/")[0])
    sel_sorted = sorted(selected_seasons, key=season_sort_key)
    latest_sel = sel_sorted[-1]

    ser_sm = pivot_time_smooth[latest_sel].fillna(0.0)
    n = len(ser_sm)

    if n >= 16:
        last8, prev8 = ser_sm.iloc[-8:].mean(), ser_sm.iloc[-16:-8].mean()
        if np.isfinite(last8) and np.isfinite(prev8) and prev8 > 0:
            delta = last8 - prev8
            pct = delta / prev8 * 100
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            insights.append(f"Latest momentum (**{latest_sel}**): {arrow} **{pct:+.1f}%** (last 8w vs prior 8w).")
            if abs(pct) < 5:
                insights.append("Trend looks **flat** (±5%) — consider a fresh stimulus or mini-block.")

    if n >= 4:
        q_last = ser_sm.iloc[-int(n/4):].mean()
        q_first = ser_sm.iloc[:int(n/4)].mean()
        if np.isfinite(q_last) and np.isfinite(q_first) and q_first > 0:
            pct_q = (q_last - q_first) / q_first * 100
            if pct_q > 10:
                insights.append("Season shows a **clear build** (>10% from early to late weeks).")
            elif pct_q < -10:
                insights.append("Season shows a **decline** (>10% down late vs early).")

    return insights

# Intensity page: HR zone distribution 
def generate_hr_zone_insights(agg_cum, pivot_season, season_sort_key):
    """Generate short text-based insights for HR zone analysis."""
    insights = []
    total_cum = agg_cum["total_time_in_zone"].sum()
    if total_cum > 0:
        top = agg_cum.loc[agg_cum["total_time_in_zone"].idxmax()]
        insights.append(
            f"- Dominant overall zone: **{top['hr_zone']}** "
            f"({top['pct_time_in_zone']:.1f}% of total time)."
        )

    if not pivot_season.empty:
        z5_season = pivot_season["Zone 5"].idxmax()
        insights.append(
            f"- Peak high-intensity exposure: **{z5_season}** "
            f"(Z5 = **{pivot_season.loc[z5_season, 'Zone 5']:.1f}%**)."
        )

        low_share = pivot_season["Zone 1"] + pivot_season["Zone 2"]
        best_low = low_share.idxmax()
        insights.append(
            f"- Best aerobic density: **{best_low}** "
            f"(Z1+Z2 = **{low_share.loc[best_low]:.1f}%**)."
        )

        sorted_seasons = sorted(pivot_season.index, key=season_sort_key)
        if len(sorted_seasons) >= 2:
            latest, prev = sorted_seasons[-1], sorted_seasons[-2]
            d_z5 = pivot_season.loc[latest, "Zone 5"] - pivot_season.loc[prev, "Zone 5"]
            d_low = (
                (pivot_season.loc[latest, "Zone 1"] + pivot_season.loc[latest, "Zone 2"])
                - (pivot_season.loc[prev, "Zone 1"] + pivot_season.loc[prev, "Zone 2"])
            )
            insights.append(
                f"- Latest vs previous: Z5 **{d_z5:+.1f}pp**, "
                f"Z1+Z2 **{d_low:+.1f}pp** (pp = percentage points)."
            )

    if not insights:
        insights.append("No sufficient data to generate insights.")

    return insights

# intensity page: section 2
def generate_weekly_intensity_insights(pivot_trend: pd.DataFrame) -> list[str]:
    """Generate simple textual insights for weekly intensity periodisation."""
    if pivot_trend.empty:
        return ["No weekly distribution data available for the selected filters."]

    total_w = len(pivot_trend)
    low_ok = (pivot_trend.get("Low", 0) >= 70).sum()
    mod_band = ((pivot_trend.get("Moderate", 0) >= 10) & (pivot_trend.get("Moderate", 0) <= 25)).sum()
    high_ok = (pivot_trend.get("High", 0) <= 10).sum()

    lines = [
        f"- Weeks aligned with **polarised targets**:",
        f"  - Low (≥70%): **{low_ok}/{total_w}**",
        f"  - Moderate (10–25%): **{mod_band}/{total_w}**",
        f"  - High (≤10%): **{high_ok}/{total_w}**",
    ]

    spikes = pivot_trend.get("High", pd.Series()).rolling(2, min_periods=1).mean()
    n_spikes = int((spikes > 12).sum())
    lines.append(f"- Detected **{n_spikes}** high-intensity spikes (>12% of weekly time). Ensure recovery microcycles follow.")

    return lines

# intensity page: section 3
def generate_seasonal_balance_insights(pivot_bal: pd.DataFrame, overall_scores: pd.DataFrame) -> list[str]:
    """Generate insights comparing actual intensity mix vs. 80/15/5 target."""
    if pivot_bal.empty:
        return ["Not enough data to evaluate seasonal balance."]

    lines = []
    ideal = {"Low": 80, "Moderate": 15, "High": 5}

    # Sport/season with highest deviation
    dev = pivot_bal.copy()
    dev["Dev_total"] = (
        abs(dev["Low"] - ideal["Low"]) +
        abs(dev["Moderate"] - ideal["Moderate"]) +
        abs(dev["High"] - ideal["High"])
    )
    worst = dev.sort_values("Dev_total", ascending=False).iloc[0]
    lines.append(f"- Largest deviation: **{worst.name[1]}** in **{worst.name[0]}** (total deviation **{worst['Dev_total']:.1f}pp**).")

    # Best and worst seasons
    if not overall_scores.empty:
        best_row = overall_scores.loc[overall_scores["Triathlon_Score"].idxmax()]
        worst_row = overall_scores.loc[overall_scores["Triathlon_Score"].idxmin()]
        lines.append(f"- Best season by overall score: **{best_row['training_season']}** (**{best_row['Triathlon_Score']:.1f}**).")
        lines.append(f"- Lowest season by overall score: **{worst_row['training_season']}** (**{worst_row['Triathlon_Score']:.1f}**).")

    return lines

# intensity page: section 4
def generate_intensity_periodisation_insights(ipi_final: pd.DataFrame) -> list[str]:
    """Generate insights for Intensity Periodisation Index (IPI)."""
    if ipi_final.empty:
        return ["No quarterly data available to score IPI."]

    lines = []
    best = ipi_final.loc[ipi_final["Intensity_Periodisation_Index"].idxmax()]
    worst = ipi_final.loc[ipi_final["Intensity_Periodisation_Index"].idxmin()]

    lines.append(f"- Best IPI: **{best['training_season']}** (**{best['Intensity_Periodisation_Index']:.0f}**).")
    lines.append(f"- Lowest IPI: **{worst['training_season']}** (**{worst['Intensity_Periodisation_Index']:.0f}**).")

    ipi_sorted = ipi_final.sort_values("training_season", key=lambda s: [season_sort_key(x) for x in s])
    if len(ipi_sorted) >= 2:
        latest, prev = ipi_sorted.iloc[-1], ipi_sorted.iloc[-2]
        d = latest["Intensity_Periodisation_Index"] - prev["Intensity_Periodisation_Index"]
        lines.append(f"- Latest vs previous IPI: **{d:+.0f}** points.")

    mean_ratios = {
        "R_12": np.nanmean(ipi_final["R_12(Q2/Q1)"].astype(float)),
        "R_23": np.nanmean(ipi_final["R_23(Q3/Q2)"].astype(float)),
        "R_34": np.nanmean(ipi_final["R_34(Q4/Q3)"].astype(float)),
    }
    weakest = min(mean_ratios, key=lambda k: (np.inf if pd.isna(mean_ratios[k]) else mean_ratios[k]))
    pretty = {"R_12": "Q1→Q2", "R_23": "Q2→Q3", "R_34": "Q3→Q4"}
    lines.append(f"- Typical limiter: **{pretty[weakest]}** build. Consider progressive overload or an extra quality session there.")

    return lines

# Balance page insights
def generate_distribution_insights(sport_totals, selected_seasons, pct_pivot):
    """Generate text insights for sport distribution trends."""
    insights = []
    if sport_totals.empty:
        return insights

    cum = sport_totals[sport_totals["training_season"].isin(selected_seasons)]
    if not cum.empty:
        cum_agg = cum.groupby("sport", as_index=False)["training_time"].sum()
        cum_total = cum_agg["training_time"].sum()
        if cum_total > 0:
            cum_agg["pct"] = cum_agg["training_time"] / cum_total * 100
            top = cum_agg.sort_values("pct", ascending=False).iloc[0]
            if len(cum_agg) > 1:
                second = cum_agg.sort_values("pct", ascending=False).iloc[1]
                gap = top["pct"] - second["pct"]
                insights.append(f"Across selected seasons, **{top['sport']}** dominates at **{top['pct']:.1f}%** (lead over next: **{gap:.1f}pp**).")
            else:
                insights.append(f"Across selected seasons, **{top['sport']}** covers the full share (100%).")

    if len(selected_seasons) >= 2:
        def season_key(s): return int(str(s).split("/")[0])
        sel_sorted = sorted(selected_seasons, key=season_key)
        shifts = []
        for sport in pct_pivot.columns:
            vals = pct_pivot.loc[sel_sorted, sport].values
            if len(vals) >= 2:
                diffs = np.diff(vals)
                max_shift = np.nanmax(np.abs(diffs)) if diffs.size else 0
                if np.isfinite(max_shift) and max_shift > 0:
                    shifts.append((sport, float(max_shift)))
        if shifts:
            sport_shift = max(shifts, key=lambda x: x[1])
            insights.append(f"Largest YoY shift: **{sport_shift[0]}** (+/− {sport_shift[1]:.1f}pp).")

    # Detect >50% domination per season
    skew_counts = []
    for s in selected_seasons:
        row = pct_pivot.loc[s]
        if row.sum() > 0:
            dom = row.idxmax()
            val = row.max()
            if val >= 50:
                skew_counts.append((s, dom, float(val)))
    if skew_counts:
        msg = "; ".join([f"{s}: **{sp} {v:.1f}%**" for s, sp, v in skew_counts])
        insights.append(f"Strong single-sport skew (≥50% share): {msg}")

    return insights

def generate_balance_insights(balance_df, all_seasons, pivot_pct, ideal):
    """Generate text insights for balance score analysis."""
    insights = []
    if balance_df.empty:
        return insights

    best = balance_df.loc[balance_df["Balance_Score"].idxmax()]
    worst = balance_df.loc[balance_df["Balance_Score"].idxmin()]
    insights.append(f"Most balanced: **{best['training_season']}** (Score {best['Balance_Score']:.1f}).")
    insights.append(f"Least balanced: **{worst['training_season']}** (Score {worst['Balance_Score']:.1f}).")

    def s_key(s): return int(str(s).split("/")[0])
    if len(all_seasons) >= 2:
        chron = sorted(all_seasons, key=s_key)
        latest, prev = chron[-1], chron[-2]
        if latest in pivot_pct.index and prev in pivot_pct.index:
            latest_row, prev_row = pivot_pct.loc[latest], pivot_pct.loc[prev]
            diffs_latest = {sp: abs(latest_row.get(sp, 0) - ideal[sp]) for sp in ideal}
            worst_sp = max(diffs_latest, key=diffs_latest.get)
            insights.append(f"**{latest}** largest gap vs ideal: **{worst_sp}** "
                            f"(Δ {latest_row.get(worst_sp, 0) - ideal[worst_sp]:+.1f}pp).")

            yoy = {sp: latest_row.get(sp, 0) - prev_row.get(sp, 0) for sp in ideal}
            yoy_big = max(yoy, key=lambda k: abs(yoy[k]))
            if abs(yoy[yoy_big]) >= 1:
                arrow = "↑" if yoy[yoy_big] > 0 else "↓"
                insights.append(f"YoY shift: **{yoy_big}** ({arrow}{abs(yoy[yoy_big]):.1f}pp).")

    return insights


