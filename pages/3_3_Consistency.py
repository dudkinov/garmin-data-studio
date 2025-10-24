# --- Imports
import streamlit as st
import pandas as pd

# --- Core data, utils, and helper modules
from core.data import ( 
    load_consistency, 
    load_long_sessions_timeline, 
    load_long_sessions_summary, 
    load_streaks,
    load_index_from_db,
    save_indexes_to_db,
    get_engine
)
from core.utils import (
    get_training_season, 
    _latest_prev, 
    apply_matplotlib_style
)
from core.tables import ( 
    compute_training_frequency_index, 
    compute_streak_penalty_index, 
    compute_streak_summary, 
    compute_long_session_summary, 
    compute_long_session_balance,
    compute_long_timeline,
    compute_distribution_consistency_index,
    compute_sub_indexes,
    combine_consistency_indexes,
    show_index_table
)
from core.visuals import (
    plot_training_consistency_distribution, 
    plot_training_frequency_bar, 
    plot_no_training_streaks, 
    plot_streak_penalty_chart, 
    plot_long_session_summary, 
    plot_long_session_balance,
    plot_distribution_consistency_index,
    plot_long_session_timeline,
    plot_consistency_index_trend
)
from core.utils import show_reconnect_button

# DB reconnect button
show_reconnect_button()

# --- Apply consistent matplotlib theme for all plots
apply_matplotlib_style()

# --- Load base datasets
df_consistency = load_consistency()
df_streaks = load_streaks()
df_long_timeline = load_long_sessions_timeline()
df_long_summary = load_long_sessions_summary()

# --- Build sport and season filters
all_sports = sorted(df_long_summary["sport"].unique())
all_training_seasons = sorted(
    set(df_consistency["training_season"])
    | set(df_streaks["training_season"])
    | set(df_long_summary["training_season"])
)

# --- Page header
st.set_page_config(layout="wide")
st.title("📊 Consistency Analysis")
st.caption("Analysis of training frequency, gaps, long sessions, and overall seasonal consistency.")

# --- Filter section
col_filters = st.columns([2, 3])
with col_filters[0]:
    sport_filter = st.radio("Sport Filter:", options=["All"] + all_sports, horizontal=True, index=0)
with col_filters[1]:
    training_season_filter = st.multiselect("Select Seasons:", options=all_training_seasons, default=all_training_seasons)
    # Ensure at least one season is always selected
    if not training_season_filter:
        training_season_filter = all_training_seasons

# --- Apply filters globally
df_consistency_f = df_consistency[df_consistency["training_season"].isin(training_season_filter)]
df_streaks_f = df_streaks[df_streaks["training_season"].isin(training_season_filter)]
df_long_summary_f = df_long_summary[df_long_summary["training_season"].isin(training_season_filter)]
df_long_timeline_f = df_long_timeline.copy()
df_long_timeline_f["start_time"] = pd.to_datetime(df_long_timeline_f["start_time"], utc=True)
df_long_timeline_f["training_season"] = df_long_timeline_f["start_time"].apply(get_training_season)
df_long_timeline_f = df_long_timeline_f[df_long_timeline_f["training_season"].isin(training_season_filter)]

if sport_filter != "All":
    df_long_summary_f = df_long_summary_f[df_long_summary_f["sport"] == sport_filter]
    df_long_timeline_f = df_long_timeline_f[df_long_timeline_f["sport"] == sport_filter]


# --- Section: Training Frequency ---
st.markdown("---")
st.subheader("📈 Training Frequency Analysis")
col1, col2 = st.columns([2, 1])

# --- Left column: frequency distribution and insights
with col1:
    pivot_df = df_consistency_f.pivot(index="training_season", columns="activities", values="days_count").fillna(0)
    if not pivot_df.empty:
        fig = plot_training_consistency_distribution(pivot_df)
        st.pyplot(fig, use_container_width=True)

        # --- Insights
        st.markdown("**🔎 Insights — Training Frequency**")
        tf_df = compute_training_frequency_index(df_consistency_f)

        if not tf_df.empty:
            best = tf_df.loc[tf_df["TF"].idxmax()]
            worst = tf_df.loc[tf_df["TF"].idxmin()]
            latest, prev = _latest_prev(tf_df["training_season"].tolist())

            st.write(
                f"- Best adherence: **{best.training_season}** with **{best.TF:.1f}%** trained days; "
                f"lowest: **{worst.training_season}** (**{worst.TF:.1f}%**)."
            )

            if latest and prev:
                curr = float(tf_df.loc[tf_df["training_season"] == latest, "TF"])
                prev_val = float(tf_df.loc[tf_df["training_season"] == prev, "TF"])
                st.write(f"- Latest vs previous: **{curr - prev_val:+.1f} pp** change in training days.")

# --- Right column: frequency index table and chart
with col2:
    freq_df = compute_training_frequency_index(df_consistency_f)

    with st.container():
        with st.expander("**🟢 Training Frequency Index (TF)**", expanded=True):
            st.caption("""
            Measures how many days per season included at least one training.  
            100 = training every day, 0 = no training days only.
            """)
            with st.expander("📉 View Chart", expanded=True):
                fig_tf = plot_training_frequency_bar(freq_df)
                st.pyplot(fig_tf, use_container_width=True)

            with st.expander("📊 View Table", expanded=False):
                st.dataframe(freq_df.style.format({"TF": "{:.1f}"}), use_container_width=True)

# --- Section: Training Gaps Analysis ---
st.markdown("---")
st.subheader("🔴 Training Gaps Analysis")
col1, col2 = st.columns([2, 1])

# --- Left column: streak visualization and summary
with col1:
    if not df_streaks_f.empty:
        fig_streaks = plot_no_training_streaks(df_streaks_f)
        st.pyplot(fig_streaks, use_container_width=True)

        # --- Insights
        st.markdown("**🔎 Insights — Gaps & Streaks**")
        streak_summary_all = compute_streak_summary(df_streaks_f)
        if not streak_summary_all.empty:
            worst = streak_summary_all.loc[streak_summary_all["max_streak"].idxmax()]
            st.write(
                f"- Longest no-training gap: **{int(worst.max_streak)} days** in **{worst.training_season}** "
                f"(avg gap {worst.avg_streak:.1f} days, {int(worst.n_streaks)} gaps)."
            )

# --- Right column: streak penalty index
with col2:
    streak_summary = compute_streak_penalty_index(df_streaks_f)

    with st.container():
        with st.expander("**🔴 Streak Penalty Index (SP)**", expanded=True):
            st.caption("Penalizes long and frequent breaks — higher = fewer interruptions in training.")
            
            with st.expander("📉 View Chart", expanded=True):
                fig_sp = plot_streak_penalty_chart(streak_summary)
                st.pyplot(fig_sp, use_container_width=True)

            with st.expander("📊 View Table", expanded=False):
                st.dataframe(
                    streak_summary[["training_season", "avg_streak", "max_streak", "SP"]]
                    .style.format({"avg_streak": "{:.1f}", "max_streak": "{:.1f}", "SP": "{:.1f}"}),
                    use_container_width=True,
                )

# --- Section: Long Session Consistency ---
st.markdown("---")
st.subheader("🟣 Long Session Consistency Analysis")
col1, col2 = st.columns([2, 1])

# --- Left column: duration, count, and insights
with col1:
    if not df_long_summary_f.empty:
        fig_long = plot_long_session_summary(df_long_summary_f)
        st.pyplot(fig_long, use_container_width=True)

        # --- Insights
        st.markdown("**🔎 Insights — Long Sessions**")
        totals = compute_long_session_summary(df_long_summary_f)

        if not totals.empty:
            top = totals.loc[totals["total_long"].idxmax()]
            st.write(
                f"- Peak long-session count: **{int(top.total_long)}** in **{top.training_season}** "
                f"(mean duration {top.avg_dur:.2f} h across sports)."
            )

            best_pair = df_long_summary_f.loc[df_long_summary_f["num_long_activities"].idxmax()]
            st.write(
                f"- Most long sessions in one sport: **{best_pair.sport} {best_pair.training_season}** "
                f"with **{int(best_pair.num_long_activities)}**."
            )

# --- Right column: long-session balance
with col2:
    lsb_summary = compute_long_session_balance(df_long_summary_f)

    with st.container():
        with st.expander("**🟣 Long-Session Balance Index (LSB)**", expanded=True):
            st.caption("""
            Represents the proportion of long sessions (cycling > 3 h, running > 1.5 h, swimming > 1.2 h)  
            relative to the most consistent season.
            """)
            with st.expander("📉 View Chart", expanded=True):
                fig_lsb = plot_long_session_balance(lsb_summary)
                st.pyplot(fig_lsb, use_container_width=True)

            with st.expander("📊 View Table", expanded=False):
                st.dataframe(lsb_summary.style.format({"LSB": "{:.1f}"}), use_container_width=True)

        # --- Insights
        if not lsb_summary.empty:
            latest, prev = _latest_prev(lsb_summary["training_season"].tolist())
            if latest:
                curr = float(lsb_summary.loc[lsb_summary["training_season"] == latest, "LSB"].values[0])
                if prev:
                    prev_val = float(lsb_summary.loc[lsb_summary["training_season"] == prev, "LSB"].values[0])
                    st.write(f"**LSB** latest vs previous: **{curr - prev_val:+.1f}** points.")

# --- Section: Long Session Distribution ---
col1, col2 = st.columns([2, 1])

# --- Left column: timeline
with col1:
    if not df_long_timeline_f.empty:
        merged = compute_long_timeline(df_long_timeline_f)
        fig_timeline = plot_long_session_timeline(merged, get_training_season)
        st.pyplot(fig_timeline, use_container_width=True)

# --- Right column: distribution consistency index
with col2:
    if not df_long_timeline_f.empty:
        dist_summary = compute_distribution_consistency_index(df_long_timeline_f, get_training_season)

        with st.container():
            with st.expander("**📅 Distribution Consistency Index (DCI)**", expanded=True):
                st.caption("Shows how well long sessions were distributed across each season.")

                with st.expander("📉 View Chart", expanded=True):
                    fig_dci = plot_distribution_consistency_index(dist_summary)
                    st.pyplot(fig_dci, use_container_width=True)

                with st.expander("📊 View Table", expanded=False):
                    st.dataframe(
                        dist_summary.sort_values("training_season", ascending=False)
                        .style.format({
                            "active_weeks": "{:.0f}",
                            "total_weeks": "{:.1f}",
                            "DCI": "{:.1f}"
                        }),
                        use_container_width=True,
                    )

            # --- Insights
            if not dist_summary.empty:
                best = dist_summary.loc[dist_summary["DCI"].idxmax()]
                worst = dist_summary.loc[dist_summary["DCI"].idxmin()]
                st.write(
                    f"- Best spread of long sessions: **{best.training_season}** (DCI **{best.DCI:.1f}**). "
                    f"Most clustered: **{worst.training_season}** (DCI **{worst.DCI:.1f}**)."
                )

# --- Section: Seasonal Consistency Index ---
st.markdown("---")
st.header("📈 Seasonal Consistency Index")

if not df_consistency_f.empty or df_streaks_f.empty or df_long_summary_f.empty or df_long_timeline_f.empty:
    # --- Load DCI2 (deload consistency index)
    df_dci2 = load_index_from_db("deload_consistency_index_2")

    # --- Compute sub-indexes
    tf, sp, ls, dci = compute_sub_indexes(
        df_consistency_f, df_streaks_f, df_long_summary_f, df_long_timeline_f, get_training_season
    )

    # --- Combine and calculate weighted final index
    combined = combine_consistency_indexes(tf, sp, ls, dci, df_dci2)

    # --- Show summary table
    styled = show_index_table(
        combined.set_index("training_season").sort_index(ascending=False),
        value_col="Consistency_Index"
    )
    st.dataframe(styled, use_container_width=True)

    # --- Charts and insights
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_ci = plot_consistency_index_trend(combined)
        st.pyplot(fig_ci, use_container_width=True)

        # --- Insights: drivers and trends
        st.markdown("**🔎 Insights — What drives the Consistency Index?**")
        latest, prev = _latest_prev(combined["training_season"].tolist())
        if latest:
            r = combined[combined["training_season"] == latest].iloc[0]
            comps = {"TF": r.TF, "SP": r.SP, "LS": r.LS, "DCI": r.DCI, "DCI2": r.DCI2}
            others = combined[combined["training_season"] != latest]
            if not others.empty:
                mean_others = {k: others[k].mean() for k in comps}
                diffs = {k: comps[k] - mean_others[k] for k in comps}
                strongest, weakest = max(diffs, key=diffs.get), min(diffs, key=diffs.get)
                st.write(
                    f"- Strength: **{strongest}** ({diffs[strongest]:+.1f} vs avg), "
                    f"focus area: **{weakest}** ({diffs[weakest]:+.1f})."
                )
            if prev:
                prev_row = combined[combined["training_season"] == prev].iloc[0]
                st.write(
                    f"- Change vs previous: TF {r.TF - prev_row.TF:+.1f}, SP {r.SP - prev_row.SP:+.1f}, "
                    f"LS {r.LS - prev_row.LS:+.1f}, DCI {r.DCI - prev_row.DCI:+.1f}, DCI₂ {r.DCI2 - prev_row.DCI2:+.1f}."
                )

    # --- Interpretation panel
    with col2:
        mean_idx = combined["Consistency_Index"].mean()
        trend = "📈 improving" if combined["Consistency_Index"].iloc[-1] > combined["Consistency_Index"].iloc[0] else "📉 declining"

        with st.expander("**🧭 Interpretation:**", expanded=True):
            st.markdown(f"""
            - Average = **{mean_idx:.1f}**
            - Highest = **{combined['Consistency_Index'].max():.1f} ({combined.loc[combined['Consistency_Index'].idxmax(), 'training_season']})**
            - Lowest = **{combined['Consistency_Index'].min():.1f} ({combined.loc[combined['Consistency_Index'].idxmin(), 'training_season']})**
            - Trend: {trend}

            **Weights**
            - TF 35%, SP 25%, LS 20%, DCI 10%, DCI₂ 10%

            **Scale**
            - 90–100 Exceptional  
            - 70–89 Stable  
            - <70 Irregular
            """)

# --- Save results to DB
st.markdown("---")
engine = get_engine()

upload_rows = []
for _, r in combined.iterrows():
    upload_rows.append({
        "training_season": str(r["training_season"]),
        "index_name": "Annual_Consistency_Index",
        "scope": "overall",
        "index_value": float(r["Consistency_Index"]),
    })

success, error = save_indexes_to_db(engine, upload_rows, "Consistency_Page")