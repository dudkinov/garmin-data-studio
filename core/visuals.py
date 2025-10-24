import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.patches as patches
from datetime import datetime
from matplotlib import colors as mcolors
from core.utils import zone_colors as zc
from core.constants import SPORT_COLORS


def plot_seasonal_hours_sessions(df):
    if df.empty:
        st.warning("No data to plot for seasonal hours & sessions.")
        return None

    seasons = df["training_season"].tolist()
    x = np.arange(len(seasons))
    w = 0.38  # bar width

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=110)

    # --- Left Y: HOURS
    bars_hours = ax.bar(
        x - w / 2,
        df["total_hours"],
        width=w,
        color="#1f77b4",
        alpha=0.85,
        label="Hours",
        zorder=2,
    )
    ax.set_ylabel("Hours / season")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    # --- Right Y: SESSIONS
    ax2 = ax.twinx()
    bars_sess = ax2.bar(
        x + w / 2,
        df["num_trainings"],
        width=w,
        color="#ff7f0e",
        alpha=0.75,
        label="Sessions",
        zorder=3,
    )
    ax2.set_ylabel("Sessions / season")

    # --- X axis
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)

    # --- Title
    ax.set_title("Total Hours & Sessions per Season", fontsize=12, fontweight="bold")

    # --- Value labels
    ymax_h = float(df["total_hours"].max()) if len(df) else 0
    ymax_s = float(df["num_trainings"].max()) if len(df) else 0

    for b in bars_hours:
        v = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + ymax_h * 0.02,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for b in bars_sess:
        v = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + ymax_s * 0.02,
            f"{int(v):,}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#ff7f0e",
        )

    # --- Combined legend
    handles = [bars_hours, bars_sess]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper left", frameon=False)

    plt.tight_layout()
    return fig

def plot_weekly_metric(pivot, ylabel, title, colors, month_ticks, month_labels, label_colors, figsize=(14,6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each season
    for season, color in colors.items():
        if season in pivot.columns:
            ax.plot(pivot.index, pivot[season], marker='o', label=f"{season} Season", color=color)
            avg_val = pivot[season].mean()
            ax.axhline(
                avg_val, color=color, linestyle='--',
                label=f"{season} Avg: {avg_val:.1f}" + ("" if ylabel == "Garmin Load" else "h")
            )

    # Month vertical lines
    for x in month_ticks:
        ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5)

    # Month labels below x-axis
    y_offset = ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for x, label, c in zip(month_ticks, month_labels, label_colors):
        ax.text(x, y_offset, label, ha='center', va='top', fontsize=9, color=c)

    # Formatting
    ax.set_xticks(range(1, pivot.index.max() + 1))
    ax.set_xlabel("Week Number", labelpad=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    return fig

SPORT_COLORS = {
    "cycling": "#1f77b4",
    "running": "#ff7f0e",
    "swimming": "#2ca02c",
    "training": "#d62728",
}

def plot_sport_distribution(data, season):
    """
    Draws a perfectly equal-size pie chart for a given season.
    Works consistently in Streamlit grids.
    """
    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=100)  # fixed size
    colors = [SPORT_COLORS.get(s, "grey") for s in data["sport"]]

    wedges, texts, autotexts = ax.pie(
        data["percent"],
        labels=None,            # hide labels on slices
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=colors,
        textprops={"fontsize": 7},
        radius=1.0              # fixed radius for all pies
    )

    # Optional: add legend outside
    ax.legend(
        data["sport"],
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=7,
        frameon=False,
    )

    ax.set_title(f"{season}", fontsize=10, pad=2)
    ax.set_aspect("equal", "box")  # ensures perfect circle
    ax.set_xlim(-1.2, 1.2)         # lock space for uniform size
    ax.set_ylim(-1.2, 1.2)
    plt.tight_layout(pad=1)

    return fig

#Trend charts
def plot_trend_metric(
    pivot,
    ylabel,
    title,
    colors,
    month_ticks,
    month_labels,
    label_colors,
    figsize=(12, 5)
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for col in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[col],
            color=colors.get(col, "black"),
            label=col,
            linewidth=2   # thicker lines for trend focus
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, rotation=45)

    # Color x-axis tick labels if desired
    for tick_label, color in zip(ax.get_xticklabels(), label_colors):
        tick_label.set_color(color)

    ax.legend()
    return fig

def plot_meta_radar_chart(meta_view, all_seasons, focus_seasons=None, season_color=None):
    """
    Plot a radar chart comparing META dimensions across training seasons.

    Parameters
    ----------
    meta_view : pd.DataFrame
        DataFrame indexed by season, with columns:
        ["Consistency", "Volume", "Intensity", "Periodisation", "Balance"]

    all_seasons : list
        List of all training seasons to plot.

    focus_seasons : list, optional
        Seasons to highlight (thicker line and higher opacity).

    season_color : callable, optional
        Function mapping (season, index) -> color (str or RGB).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated radar chart figure (ready for Streamlit or saving).
    """
    if focus_seasons is None:
        focus_seasons = []
    if season_color is None:
        # Default fallback palette
        def season_color(season, i):
            cmap = plt.get_cmap("tab10")
            return cmap(i % 10)

    # Labels for the axes
    labels = ["Consistency", "Volume", "Intensity", "Periodisation", "Balance"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # --- Initialize radar chart
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 7))

    # --- Draw one line per season
    for i, season in enumerate(all_seasons):
        vals = [meta_view.loc[season, l] if pd.notna(meta_view.loc[season, l]) else 0 for l in labels]
        vals += vals[:1]
        col = season_color(season, i)
        is_focus = season in focus_seasons
        lw     = 2.2 if is_focus else 1.0
        a_line = 1.0 if is_focus else 0.25
        a_fill = 0.15 if is_focus else 0.04

        ax.plot(angles, vals, linewidth=lw, label=season, color=col, alpha=a_line)
        ax.fill(angles, vals, color=col, alpha=a_fill)

    # --- Styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.set_title("META Dimensions per Training Season (Overall)", pad=20, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(4, len(all_seasons)))

    return fig


# --- Helper to plot HR zones chart ---
def plot_zone_distribution(data, title, highlight=False):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    zone_colors_dict = zc() if callable(zc) else zc
    colors = data["hr_zone"].map(zone_colors_dict)
    ax.bar(data["hr_zone"], data["pct_time_in_zone"], color=colors)

    ax.set_title(title, fontsize=11, fontweight="bold" if highlight else "normal")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%", fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    for idx, val in enumerate(data["pct_time_in_zone"]):
        ax.text(idx, val + 1, f"{val}%", ha="center", fontsize=7)

    if highlight:
        ax.set_facecolor("#e8e8e8")

    return fig

def draw_radar_chart(meta_df, seasons, focus="All"):
    labels = meta_df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 7))

    for i, s in enumerate(seasons):
        vals = meta_df.loc[s].fillna(0).tolist()
        vals += vals[:1]
        color = plt.cm.tab10(i % 10)
        lw, a_line, a_fill = (2.2, 1.0, 0.15) if focus in (s, "All") else (1.0, 0.25, 0.04)
        ax.plot(angles, vals, color=color, linewidth=lw, label=s, alpha=a_line)
        ax.fill(angles, vals, color=color, alpha=a_fill)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.set_title("Summary Season Dimensions", pad=20, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=min(4, len(seasons)))
    st.pyplot(fig, use_container_width=True)

#Index charts, volume page
def plot_deload_consistency(dci_df):
    """Create bar chart for Deload Consistency Index (DCI₂)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dci_df))
    vals = dci_df["DCI₂"]
    ax.bar(x, vals, color="#FFD700", alpha=0.9, width=0.6)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(dci_df["training_season"])
    ax.set_ylabel("Index (0–100)")
    ax.set_title("Deload Consistency per Season", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig

#Periodisation Index Chart, Volume Page
def plot_svpi_bar_chart(svpi_df: pd.DataFrame):
    """Draw bar chart for Season Volume Periodisation Index (SVPI)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(svpi_df))
    vals = svpi_df["SVPI"]

    ax.bar(x, vals, color="#4169E1", alpha=0.9, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(svpi_df["training_season"])
    ax.set_ylabel("Index (0–100)")
    ax.set_title("Season Volume Periodisation per Season")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig


# Consistency page, section 1: frequency chart
def plot_training_consistency_distribution(pivot_df):
    """Bar chart showing % of trained vs non-trained days per season."""
    if pivot_df.empty:
        return None

    pivot_df["total_days"] = pivot_df.sum(axis=1)
    pivot_df["no_training_days"] = pivot_df.get(0, 0)
    pivot_df["trained_days"] = pivot_df["total_days"] - pivot_df["no_training_days"]

    pivot_df_pct = (
        pivot_df.div(pivot_df["total_days"], axis=0)
        .replace([np.inf, -np.inf], 0)
        .fillna(0) * 100
    )

    train_cols = [c for c in pivot_df.columns if isinstance(c, (int, float, np.integer)) and c > 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pivot_df.index))
    bar_width = 0.6
    bottom = np.zeros(len(pivot_df))

    # No-training bar
    ax.bar(
        x - bar_width / 4,
        pivot_df_pct["no_training_days"],
        width=bar_width / 2,
        color="gray",
        label="No Trainings"
    )

    # Stacked trained bars
    colors = plt.cm.tab10.colors
    for i, c in enumerate(train_cols):
        ax.bar(
            x + bar_width / 4,
            pivot_df_pct[c],
            width=bar_width / 2,
            bottom=bottom,
            color=colors[i % len(colors)],
            label=f"{int(c)} trainings/day"
        )
        bottom += pivot_df_pct[c].values

    # Labels
    for i, _ in enumerate(pivot_df.index):
        no_train_pct = pivot_df_pct["no_training_days"].iloc[i]
        trained_pct = 100 - no_train_pct
        ax.text(x[i] - bar_width / 4, no_train_pct + 1, f"{no_train_pct:.1f}%", ha="center", fontsize=8)
        ax.text(x[i] + bar_width / 4, trained_pct + 1, f"{trained_pct:.1f}%", ha="center", fontsize=8)

    ax.set_title("Training Consistency per Season: Trained vs No Training", pad=15)
    ax.set_xlabel("Training Season")
    ax.set_ylabel("Percentage of Days")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index.astype(str))
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    ax.legend(title="Daily Activities", frameon=True, edgecolor="lightgray", loc="upper left")

    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.1)
    return fig

def plot_training_frequency_bar(freq_df):
    """Plot Training Frequency Index per season as a bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = freq_df["training_season"]
    x = np.arange(len(x_labels))
    ax.bar(x, freq_df["TF"], color="#1f77b4", width=0.6, alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_title("Training Frequency per Season", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Season")
    ax.set_ylabel("Frequency Index (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    # Annotate bars
    for i, v in enumerate(freq_df["TF"]):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    return fig

def plot_no_training_streaks(df_streaks):
    """Plot no-training streaks by season with compressed y-scale."""
    if df_streaks.empty:
        return None

    def color_by_length(x):
        if x <= 2: return "royalblue"
        elif x <= 5: return "green"
        elif x <= 10: return "orange"
        return "red"

    def compress_height(x, threshold=15):
        return x if x <= threshold else threshold + np.log1p(x - threshold) * 2

    df = df_streaks.copy()
    df["color"] = df["streak_length"].apply(color_by_length)
    df["compressed_height"] = df["streak_length"].apply(compress_height)

    season_labels = sorted(df["training_season"].unique())
    x_positions = []
    for i, season in enumerate(season_labels):
        streaks = df[df["training_season"] == season]
        positions = np.linspace(i - 0.4, i + 0.4, len(streaks))
        x_positions.extend(positions)
    df["x_pos"] = x_positions

    ytick_real = [0, 5, 10, 15, 20, 30, 40, 50]
    ytick_display = [compress_height(y) for y in ytick_real]

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.bar(row["x_pos"], row["compressed_height"], color=row["color"], width=0.3, alpha=0.9)

    ax.set_title("No-Training Streaks by Season", pad=15)
    ax.set_xlabel("Training Season")
    ax.set_ylabel("Streak Duration (days, compressed scale)")
    ax.set_xticks(range(len(season_labels)))
    ax.set_xticklabels(season_labels, rotation=45)
    ax.set_yticks(ytick_display)
    ax.set_yticklabels([str(y) for y in ytick_real])
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    legend_elements = [
        Patch(facecolor="royalblue", label="1–2 days"),
        Patch(facecolor="green", label="3–5 days"),
        Patch(facecolor="orange", label="6–10 days"),
        Patch(facecolor="red", label=">10 days")
    ]
    ax.legend(
        handles=legend_elements,
        title="Streak Duration (days)",
        frameon=True,
        edgecolor="lightgray",
        loc="upper left",
        facecolor="white"
    )

    plt.tight_layout()
    return fig


def plot_streak_penalty_chart(streak_summary):
    """Plot Streak Penalty Index (SP) per season."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = streak_summary["training_season"]
    x = np.arange(len(x_labels))
    ax.bar(x, streak_summary["SP"], color="#ff7f0e", width=0.6, alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_title("No-Training Streak Penalty by Season", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Season")
    ax.set_ylabel("Penalty Index (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    for i, v in enumerate(streak_summary["SP"]):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    return fig

#Consistency page: section 3
def plot_long_session_summary(df_long_summary):
    """Plot average duration and number of long activities per sport & season."""
    if df_long_summary.empty:
        return None

    sport_colors = {"cycling": "tab:blue", "running": "tab:orange", "swimming": "tab:green"}
    visible_sports = df_long_summary["sport"].unique()
    x_labels = sorted(df_long_summary["training_season"].unique())
    x = np.arange(len(x_labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Chart 1 — Average Duration
    ax = axes[0]
    for i, sport in enumerate(visible_sports):
        data = df_long_summary[df_long_summary["sport"] == sport]
        values = data.set_index("training_season").reindex(x_labels)["avg_duration"].fillna(0)
        ax.bar(
            x + i * width - (len(visible_sports) - 1) * width / 2,
            values,
            width=width,
            color=sport_colors.get(sport, "gray"),
            label=sport,
        )
    ax.set_title("Average Duration (hours)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Hours")
    ax.set_xlabel("Training Season")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    # Chart 2 — Number of Long Activities
    ax = axes[1]
    for i, sport in enumerate(visible_sports):
        data = df_long_summary[df_long_summary["sport"] == sport]
        values = data.set_index("training_season").reindex(x_labels)["num_long_activities"].fillna(0)
        ax.bar(
            x + i * width - (len(visible_sports) - 1) * width / 2,
            values,
            width=width,
            color=sport_colors.get(sport, "gray"),
            label=sport,
        )
    ax.set_title("Number of Long Activities", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Training Season")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    # Unified legend
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, title="Sport", loc="upper left")
    plt.tight_layout()
    return fig


def plot_long_session_balance(lsb_summary):
    """Plot Long-Session Balance (LSB) index."""
    if lsb_summary.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = lsb_summary["training_season"]
    x = np.arange(len(x_labels))
    ax.bar(x, lsb_summary["LSB"], width=0.6, alpha=0.9, color="tab:purple")
    ax.set_ylim(0, 100)
    ax.set_title("Long-Session Balance per Season", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Season")
    ax.set_ylabel("Balance Index (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    for i, v in enumerate(lsb_summary["LSB"]):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig

#Consistency page: section 4
def plot_long_session_timeline(merged, get_training_season):
    """Plot long-session timeline distribution across all days."""
    if merged.empty:
        return None

    merged["training_season"] = merged["date"].apply(get_training_season)
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["x_pos"] = np.arange(len(merged))

    fig, ax = plt.subplots(figsize=(14, 6))
    for sport, sport_df in merged.groupby("sport"):
        ax.bar(
            sport_df["x_pos"],
            sport_df["avg_duration"],
            color=sport_df["color"].iloc[0],
            width=1.8,
            alpha=0.85,
            linewidth=0,
        )

    training_seasons = sorted(merged["training_season"].unique())
    year_ticks = [merged.loc[merged["training_season"] == y, "x_pos"].median() for y in training_seasons]

    ax.set_xticks(year_ticks)
    ax.set_xticklabels(training_seasons)
    ax.set_xlabel("Training Season")
    ax.set_ylabel("Average Duration (hours)")
    ax.set_title("Long Training Sessions Distribution Across All Days", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    legend_elements = [
        Patch(facecolor="tab:blue", label="Cycling"),
        Patch(facecolor="tab:orange", label="Running"),
        Patch(facecolor="tab:green", label="Swimming"),
    ]
    ax.legend(handles=legend_elements, title="Sport", loc="upper left")

    plt.tight_layout()
    return fig

def plot_distribution_consistency_index(dist_summary):
    """Plot Distribution Consistency Index (DCI) bar chart."""
    if dist_summary.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = dist_summary["training_season"]
    x = np.arange(len(x_labels))

    ax.bar(x, dist_summary["DCI"], color="#9467bd", width=0.6, alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_title("Distribution Consistency Index (DCI) per Season", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Season")
    ax.set_ylabel("DCI (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    for i, v in enumerate(dist_summary["DCI"]):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig

#Consistency page: fanal index plot
def plot_consistency_index_trend(df):
    """Bar + line chart for Consistency Index trend."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = df["training_season"]
    x = np.arange(len(x_labels))
    values = df["Consistency_Index"]

    ax.bar(x, values, color="royalblue", width=0.6, alpha=0.9)
    ax.plot(x, values, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_ylim(0, 100)
    ax.set_title("Seasonal Consistency Index Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Season")
    ax.set_ylabel("Index (0–100)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    for i, v in enumerate(values):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig

#Intensity page: HR distribution plot
def plot_seasonal_zone_distribution(pivot_df, selected_sport, zone_colors):
    """Render per-season HR zone bar chart."""
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=110)
    x = np.arange(len(pivot_df.index))
    width = 0.16

    for i, z in enumerate(pivot_df.columns):
        bars = ax.bar(
            x + (i - (len(pivot_df.columns)-1)/2) * width,
            pivot_df[z],
            width=width,
            color=zone_colors[z],
            label=z,
            alpha=0.9
        )
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    ax.set_ylabel("% of time")
    ax.set_title(
        f"Seasonal HR Zone Distribution — {selected_sport if selected_sport!='Overall' else 'All Sports'}",
        fontsize=12,
        fontweight="bold"
    )
    ax.legend(ncol=5, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig

#Intensity page: section 2
def plot_weekly_intensity(pivot_trend: pd.DataFrame):
    """Return a Matplotlib Figure with weekly stacked intensity distribution."""
    if pivot_trend.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Create the figure and axes explicitly (prevents Streamlit warning)
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=110)

    # Plot the stacked intensity areas
    ax.stackplot(
        pivot_trend.index,
        pivot_trend.get("Low", pd.Series(0, index=pivot_trend.index)),
        pivot_trend.get("Moderate", pd.Series(0, index=pivot_trend.index)),
        pivot_trend.get("High", pd.Series(0, index=pivot_trend.index)),
        labels=["Low (Z1+Z2)", "Moderate (Z3+Z4)", "High (Z5)"],
        colors=["#1f77b4", "#ff7f0e", "#d62728"],
        alpha=0.85
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("% of time")
    ax.set_title("Weekly Intensity Distribution", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=False, ncol=3)
    ax.grid(alpha=0.3)

    return fig

# Balance page: cumulative chart
def plot_cumulative_pie(df_in, selected_seasons):
    selected = df_in if not selected_seasons else df_in[df_in["training_season"].isin(selected_seasons)]
    if selected.empty:
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    agg = selected.groupby("sport", as_index=False)["training_time"].sum()
    total = agg["training_time"].sum()
    if total <= 0:
        agg["percent"] = 0.0
    else:
        agg["percent"] = agg["training_time"] / total * 100

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
    colors = [SPORT_COLORS.get(s, "grey") for s in agg["sport"]]
    ax.pie(
        agg["percent"],
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        startangle=90,
        counterclock=False,
        colors=colors,
        textprops={"fontsize": 8},
        radius=1.0,
    )
    ax.legend(
        agg["sport"],
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=7,
        frameon=False,
    )
    ax.set_title("Cumulative (selected seasons)", fontsize=10, pad=2)
    ax.set_aspect("equal", "box")
    plt.tight_layout(pad=1)
    return fig

# Balance page: time distribution per season
def plot_stacked_distribution(df_in, selected_seasons):
    df_filtered = df_in[df_in["training_season"].isin(selected_seasons)]
    if df_filtered.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    pivot = (
        df_filtered.pivot_table(index="training_season", columns="sport", values="training_time", aggfunc="sum")
        .fillna(0)
        .sort_index()
    )
    row_sums = pivot.sum(axis=1)
    pivot = pivot.div(row_sums.replace({0: np.nan}), axis=0) * 100
    pivot = pivot.fillna(0)

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    bottom = np.zeros(len(pivot))

    for sport, color in SPORT_COLORS.items():
        if sport in pivot.columns:
            values = pivot[sport].values
            ax.bar(
                pivot.index,
                values,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

            # inline labels on bigger segments for readability
            for i, (y, b) in enumerate(zip(values, bottom)):
                if y > 8:  # show label only when big enough
                    text_color = "white" if color not in ("#ff7f0e", "#f0c36e") else "black"
                    ax.text(
                        i,
                        b + y / 2,
                        f"{sport}\n{y:.1f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7.5,
                        fontweight="bold",
                    )
            bottom += values

    ax.set_title("Share of Training Time by Sport per Season", fontsize=11, pad=8)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    return fig

#Balance page: index chart
def plot_balance_bar(balance_df: pd.DataFrame):
    """Draws a bar chart showing balance score per season."""
    if balance_df.empty or "Balance_Score" not in balance_df.columns:
        st.info("No balance data available to visualize.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(balance_df["training_season"], balance_df["Balance_Score"], color="#2ca02c", alpha=0.85)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Balance Score (0–100)")
    ax.set_title("Discipline Balance per Season", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    # Add labels above bars
    for i, v in enumerate(balance_df["Balance_Score"]):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    st.pyplot(fig, use_container_width=True)

# Plan and track: plan vs actual with tolerance band
def plot_plan_vs_real(plan_series, band_low, band_high, real_series, month_ticks, month_labels):
    """Plot weekly training plan vs real data, keeping real line flat after last recorded week."""
    # Extend the real series horizontally (flat) after last known point
    real_extended = real_series.copy()
    last_valid = real_series[real_series > 0].last_valid_index()
    if last_valid and last_valid < len(real_series):
        flat_value = real_series.loc[last_valid]
        real_extended.loc[last_valid+1:] = flat_value

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=120)
    ax.plot(plan_series.index, plan_series.values, label="Plan (macro × micro)", linewidth=2.6, color="#1f77b4")
    ax.fill_between(plan_series.index, band_low, band_high, alpha=0.16, color="#1f77b4", label="±10% band")
    ax.plot(real_extended.index, real_extended.values, label="Real 25/26", linewidth=2, color="#ff7f0e")
    
    # Shade deload weeks (2, 6, 10, ...)
    for w in range(2, 53, 4):
        ax.axvspan(w - 0.5, w + 0.5, color="#ddd", alpha=0.35, lw=0)

    ax.set_xlabel("Season week")
    ax.set_ylabel("Hours / week")
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.set_ylim(0, max(plan_series.max(), real_series.max()) * 1.25)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Weekly Training Time — Plan (with 4-week microcycles) vs Real", fontsize=12, fontweight="bold")
    st.pyplot(fig, use_container_width=True)

# Plan and track: cumulative
def plot_cumulative_progress(plan_cum, real_cum, month_ticks, month_labels, cur_week=None, on_track=None):
    """Plot cumulative training load (plan vs actual)."""
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=120)
    ax.plot(plan_cum.index, plan_cum.values, linewidth=2.4, label="Plan (cum.)", color="#1f77b4")
    ax.plot(real_cum.index, real_cum.values, linewidth=2.0, label="Real (cum.)", color="#ff7f0e")

    if cur_week and cur_week in real_cum.index:
        ax.scatter([cur_week], [real_cum.loc[cur_week]], s=25, color="#ff7f0e", zorder=3)

    ax.set_xlabel("Season week")
    ax.set_ylabel("Cumulative Hours")
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    title = f"Cumulative Hours — On-track: {on_track:.1f}%" if np.isfinite(on_track) else "Cumulative Hours"
    ax.set_title(title, fontsize=12, fontweight="bold")
    st.pyplot(fig, use_container_width=True)


def plot_microcycle_bars(plan_series, month_ticks, month_labels):
    """Plot planned weekly hours with deloads shaded."""
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=120)
    ax.bar(plan_series.index, plan_series.values, width=0.9, color="#1f77b4", alpha=0.85)
    for w in range(2, 53, 4):  # deload bars: 2, 6, 10, ...
        if w in plan_series.index:
            ax.bar(w, plan_series.loc[w], width=0.9, color="#9aa5b1", alpha=0.95)
    ax.set_xlabel("Season week")
    ax.set_ylabel("Planned Hours")
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.set_title("Planned Weekly Hours with 4-Week Microcycles (deloads shaded)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)


def plot_execution_heatmap(plan_series, real_series):
    """Plot Real/Plan execution ratio as a 13x4 microcycle heatmap with current week highlighted."""
    # Compute real/plan ratio safely
    ratio = (real_series / plan_series).replace([np.inf, -np.inf], np.nan).fillna(0).values

    # Ensure 13×4 structure (52 weeks)
    n_cycles = 13
    padded_len = n_cycles * 4
    if len(ratio) < padded_len:
        ratio = np.pad(ratio, (0, padded_len - len(ratio)), constant_values=np.nan)

    # Shift forward by 2 weeks (aligns microcycle anchor)
    shift_weeks = 2
    ratio_shifted = np.roll(ratio, shift_weeks)
    ratio_shifted[:shift_weeks] = np.nan
    mat = ratio_shifted.reshape(n_cycles, 4)

    # Compute current week index (based on calendar)
    today = datetime.now()
    season_start = datetime(today.year if today.month >= 9 else today.year - 1, 9, 1)
    week_index = int((today - season_start).days // 7)
    week_index = min(max(0, week_index), n_cycles * 4 - 1)

    # Apply same 2-week shift
    current_idx_shifted = (week_index + shift_weeks) % (n_cycles * 4)
    cur_microcycle = current_idx_shifted // 4
    cur_week_in_cycle = current_idx_shifted % 4

    # Draw heatmap
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=120)
    norm = mcolors.TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.4)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_title("Execution Heatmap — Real / Plan per Microcycle", fontsize=12, fontweight="bold")
    ax.set_xlabel("Week in microcycle (1..4)")
    ax.set_ylabel("Microcycle # (Sep→Aug)")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_yticks(np.arange(n_cycles))
    ax.set_yticklabels([str(i + 1) for i in range(n_cycles)])

    # Subtle grid lines
    ax.set_yticks(np.arange(mat.shape[0]) - 0.5, minor=True)
    ax.set_xticks(np.arange(mat.shape[1]) - 0.5, minor=True)
    ax.grid(which="minor", color="#e5e7eb", linestyle="--", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Text labels inside cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if (val < 0.75 or val > 1.25) else "black"
            ax.text(j, i, f"{val * 100:,.0f}%", ha="center", va="center", fontsize=8, color=txt_color)

    # Highlight current microcycle/week
    rect = patches.Rectangle(
        (cur_week_in_cycle - 0.5, cur_microcycle - 0.5),
        1, 1,
        linewidth=2.5,
        edgecolor="orange",
        facecolor="none"
    )
    ax.add_patch(rect)

    # Colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Real / Plan")
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

# 7 page: sport time distribution chart  
def plot_sport_pie(df_in, title):
    """Render sport distribution pie chart (thread-safe for Streamlit)."""
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=110)
    colors = [SPORT_COLORS.get(s, "grey") for s in df_in["sport"]]

    ax.pie(
        df_in["percent"],
        labels=df_in["sport"],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=colors,
        textprops={"fontsize": 9},
    )
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_aspect("equal", "box")

    return fig

# Page 7: time in zone chart

def plot_weekly_hr_zones(pv, zone_order, zone_colors, month_ticks, month_labels):
    """Render stacked area chart for weekly time in HR zones."""
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=120)
    y_stack = [pv[z].values for z in zone_order]

    ax.stackplot(
        pv["week_index"].values,
        y_stack,
        labels=zone_order,
        colors=[zone_colors[z] for z in zone_order],
        alpha=0.95,
    )

    ax.set_xlabel("Season week")
    ax.set_ylabel("Hours / week")
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.set_title("Weekly Time in HR Zones (stacked)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", ncol=5, frameon=False)
    return fig


def plot_zone_distribution_bars(comp, zone_order):
    """Render horizontal bar chart comparing planned vs real HR zone mix."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=120)
    y = np.arange(len(zone_order))

    ax.barh(y - 0.18, comp["plan_pct"], height=0.35, label="Planned", color="#1f77b4", alpha=0.9)
    ax.barh(y + 0.18, comp["real_pct"], height=0.35, label="Real YTD", color="#ff7f0e", alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(zone_order)
    ax.set_xlabel("% of time")
    ax.set_xlim(0, max(100, (comp[["plan_pct", "real_pct"]].to_numpy().max() + 5)))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Zone Mix — Planned vs Real", fontsize=12, fontweight="bold")

    # Labels
    for i, (p, r) in enumerate(zip(comp["plan_pct"], comp["real_pct"])):
        ax.text(max(p, r) + 1, i - 0.18 if p >= r else i + 0.18,
                f"{p:.1f}% / {r:.1f}%", va="center", fontsize=8)

    return fig