import pandas as pd
import streamlit as st
from math import isfinite
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy import text
import numpy as np
from datetime import datetime
from sqlalchemy.exc import OperationalError
import time
import os

# --- DB connection
@st.cache_resource
def get_engine():
    """
    Create and cache a resilient SQLAlchemy engine.
    Automatically adds SSL, pings connections before use,
    and recycles idle connections.
    """
    load_dotenv()

    # Prefer Streamlit secrets; fall back to .env if running locally
    db_url = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("❌ DATABASE_URL not found")

    # Enforce SSL requirement if missing
    if "sslmode" not in db_url:
        db_url += "&sslmode=require" if "?" in db_url else "?sslmode=require"

    # Create resilient engine
    engine = create_engine(
        db_url,
        pool_pre_ping=True,     
        pool_recycle=180,       
        future=True
    )
    return engine


# --- Connection refresher
def safe_query(query: str, engine, params: dict | None = None, retries: int = 2) -> pd.DataFrame:
    """
    Execute SQL safely with automatic reconnection and DataFrame output.
    - Retries once if Neon closed the SSL connection.
    - Returns an empty DataFrame on failure instead of crashing Streamlit.
    """
    for attempt in range(retries):
        try:
            with engine.connect() as conn:
                stmt = text(query)
                result = conn.execute(stmt, params or {})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df

        except OperationalError as e:
            # Handle Neon "SSL connection closed" after idle sleep
            if "SSL connection has been closed" in str(e) and attempt < retries - 1:
                engine.dispose()   # drop broken connections
                time.sleep(2)      # small delay before retry
                continue
            st.warning(f"⚠️ Database connection lost: {e}")
            return pd.DataFrame()

        except Exception as e:
            st.error(f"❌ Query failed: {e}")
            engine.dispose()
            return pd.DataFrame()

# Helper for write operations
def safe_execute(engine, query: str, params: dict | None = None) -> str:
    """Execute write queries safely with automatic reconnect."""
    try:
        with engine.begin() as conn:
            conn.execute(text(query), params or {})
        return "✅ Saved successfully."
    except OperationalError as e:
        engine.dispose()
        return f"⚠️ Connection dropped, please retry: {e}"
    except Exception as e:
        engine.dispose()
        return f"❌ Write failed: {e}"

# Load data, totals. Clean week start, add training season
@st.cache_data(show_spinner=True)
def load_data():
    engine = get_engine()
    query = """
    SELECT 
        DATE_TRUNC('week', start_time) AS week_start,
        ROUND(SUM(total_timer_time::numeric / 3600), 2) AS training_time
    FROM session
    WHERE sport IN ('running', 'cycling', 'swimming', 'training')
    GROUP BY week_start
    ORDER BY week_start;
    """
    df = safe_query(query, engine)
    df["training_time"] = df["training_time"].astype(float)
    df["week_start"] = pd.to_datetime(df["week_start"], utc=True).dt.tz_localize(None).dt.floor("D")
    df['training_season'] = pd.cut(
    df['week_start'],
    bins=[
        pd.Timestamp('2020-09-01'),
        pd.Timestamp('2021-09-01'),
        pd.Timestamp('2022-09-01'),
        pd.Timestamp('2023-09-01'),
        pd.Timestamp('2024-09-01'),
        pd.Timestamp('2025-09-01'),
        pd.Timestamp('2026-09-01')
    ],
    labels=['2020/21', '2021/22', '2022/23', '2023/24','2024/25','2025/26'],
    right=False
    )
    return df

# Load per-sport data. Clean week start, add training season
@st.cache_data(show_spinner=True)
def load_sport_data():
    engine = get_engine()
    query = """
    SELECT 
        DATE_TRUNC('week', start_time) AS week_start,
        ROUND(SUM(total_timer_time::numeric / 3600), 2) AS training_time,
        sport
    FROM session
    WHERE sport IN ('running', 'cycling', 'swimming', 'training')
    GROUP BY week_start, sport
    ORDER BY week_start, sport;
    """
    df = safe_query(query, engine)
    df["training_time"] = df["training_time"].astype(float)
    df["week_start"] = pd.to_datetime(df["week_start"], utc=True).dt.tz_localize(None).dt.floor("D")
    df['training_season'] = pd.cut(
    df['week_start'],
    bins=[
        pd.Timestamp('2020-09-01'),
        pd.Timestamp('2021-09-01'),
        pd.Timestamp('2022-09-01'),
        pd.Timestamp('2023-09-01'),
        pd.Timestamp('2024-09-01'),
        pd.Timestamp('2025-09-01'),
        pd.Timestamp('2026-09-01')
    ],
    labels=['2020/21', '2021/22', '2022/23', '2023/24','2024/25','2025/26'],
    right=False
    )
    return df

@st.cache_data(show_spinner=True)
def load_scope_data():
    # training season = Sep–Aug. Also compute the *actual* number of weeks per season.
    query = """
    WITH base AS (
        SELECT
            CASE
                WHEN EXTRACT(MONTH FROM s.start_time) >= 9 THEN 
                    CONCAT(EXTRACT(YEAR FROM s.start_time)::text,'/',LPAD(((EXTRACT(YEAR FROM s.start_time)+1)::int % 100)::text,2,'0'))
                ELSE 
                    CONCAT((EXTRACT(YEAR FROM s.start_time)::int-1)::text,'/',LPAD((EXTRACT(YEAR FROM s.start_time)::int % 100)::text,2,'0'))
            END AS training_season,
            DATE_TRUNC('week', s.start_time) AS wk,
            s.total_timer_time,
            s.sport
        FROM session s
        WHERE s.sport IN ('cycling','running','swimming','training')
          AND s.start_time < '2025-09-01'
    )
    SELECT
        training_season,
        COUNT(*) AS num_trainings,
        ROUND(SUM(total_timer_time::numeric)/3600.0, 2) AS total_hours,
        COUNT(DISTINCT wk) AS weeks_in_season
    FROM base
    GROUP BY training_season
    ORDER BY training_season;
    """
    engine = get_engine()
    df = safe_query(query, engine)
    return df

@st.cache_data
def load_streaks():
    engine = get_engine()
    query = """
    WITH daily AS (
        SELECT 
            DATE_TRUNC('day', c.calendar_date) AS day,
            CASE
                WHEN EXTRACT(MONTH FROM c.calendar_date) >= 9 THEN 
                    CONCAT(EXTRACT(YEAR FROM c.calendar_date)::text,'/',LPAD(((EXTRACT(YEAR FROM c.calendar_date)+1)::int%100)::text,2,'0'))
                ELSE 
                    CONCAT((EXTRACT(YEAR FROM c.calendar_date)::int - 1)::text,'/',LPAD((EXTRACT(YEAR FROM c.calendar_date)::int%100)::text,2,'0'))
            END AS training_season,
            COUNT(s.file_uid) AS activities,
            CASE WHEN COUNT(s.file_uid) > 0 THEN 1 ELSE 0 END AS trained
        FROM (
            SELECT generate_series('2021-02-19'::date, '2025-08-31'::date, '1 day') AS calendar_date
        ) c
        LEFT JOIN session s
            ON DATE(s.start_time) = c.calendar_date AND s.sport IN ('cycling','running','swimming','training')
        GROUP BY c.calendar_date
        HAVING c.calendar_date < '2025-09-01'
    ),
    marked AS (
        SELECT
            day, training_season, activities, trained,
            SUM(trained) OVER (ORDER BY day) AS training_counter
        FROM daily
    ),
    streaks AS (
        SELECT MIN(training_season) AS training_season, training_counter, COUNT(*) AS streak_length
        FROM marked WHERE trained = 0
        GROUP BY training_counter
    )
    SELECT training_season, training_counter, streak_length
    FROM streaks WHERE streak_length > 0
    ORDER BY training_season, training_counter;
    """
    engine = get_engine()
    df = safe_query(query, engine)
    return df

@st.cache_data
def load_long_sessions_timeline():
    query = """
    SELECT start_time, sport,
           ROUND(AVG(total_timer_time::numeric / 3600), 2) AS avg_duration,
           COUNT(*) AS num_long_activities
    FROM public.session
    WHERE (sport='cycling' AND total_timer_time>10800)
       OR (sport='running' AND total_timer_time>5400)
       OR (sport='swimming' AND total_timer_time>4300)
    GROUP BY sport, start_time
    ORDER BY start_time;
    """
    engine = get_engine()
    df = safe_query(query, engine)
    return df

@st.cache_data
def load_long_sessions_summary():
    query = """
    SELECT
        CASE
            WHEN EXTRACT(MONTH FROM start_time) >= 9 THEN 
                CONCAT(EXTRACT(YEAR FROM start_time)::text,'/',LPAD(((EXTRACT(YEAR FROM start_time)+1)::int%100)::text,2,'0'))
            ELSE 
                CONCAT((EXTRACT(YEAR FROM start_time)::int-1)::text,'/',LPAD((EXTRACT(YEAR FROM start_time)::int%100)::text,2,'0'))
        END AS training_season,
        sport,
        ROUND(AVG(total_timer_time::numeric / 3600), 2) AS avg_duration,
        COUNT(*) AS num_long_activities
    FROM public.session
    WHERE ((sport='cycling' AND total_timer_time>10800)
        OR (sport='running' AND total_timer_time>5400)
        OR (sport='swimming' AND total_timer_time>4300))
      AND start_time < '2025-09-01'
    GROUP BY sport, training_season
    ORDER BY sport, training_season DESC;
    """
    engine = get_engine()
    df = safe_query(query, engine)
    return df

@st.cache_data
def load_consistency():
    query = """
        WITH daily_counts AS (
        SELECT DATE_TRUNC('day', c.calendar_date) AS day,
               COUNT(s.file_uid) AS activities
        FROM (
            SELECT generate_series('2021-02-19'::date, '2025-08-31'::date , '1 day') AS calendar_date
        ) c
        LEFT JOIN session s
          ON DATE(s.start_time) = c.calendar_date and s.sport IN ('cycling','running','swimming','training')
        GROUP BY c.calendar_date
        HAVING c.calendar_date < '2025-09-01'
    )
    SELECT 
        CASE
            WHEN EXTRACT(MONTH FROM day) >= 9 THEN 
                CONCAT(EXTRACT(YEAR FROM day)::text,
                       '/',
                       LPAD(((EXTRACT(YEAR FROM day) + 1)::int % 100)::text, 2, '0'))
            ELSE 
                CONCAT(
                    (EXTRACT(YEAR FROM day)::int - 1)::text,
                    '/',
                    LPAD((EXTRACT(YEAR FROM day)::int % 100)::text, 2, '0')
                )
        END AS training_season,
        activities,
        COUNT(*) AS days_count
    FROM daily_counts
    GROUP BY training_season, activities
    ORDER BY training_season, activities;
    """
    engine = get_engine()
    df = safe_query(query, engine)
    return df

@st.cache_data(show_spinner=True)
def load_hr_zone_data_weekly():
    query = """
WITH intervals AS (
    SELECT
        s.sport,
        DATE_TRUNC('week', s.start_time) AS week_start,
        z.zone_name,
        h.seconds_in_zone / 3600.0 AS hours_in_zone
    FROM hr_zone_summary h
    JOIN session s
      ON h.file_uid = s.file_uid
     AND h.sport = s.sport
    JOIN hr_zones z
      ON h.sport = z.sport
     AND h.zone_name = z.zone_name
    WHERE s.sport IN ('cycling', 'running', 'swimming', 'training')
      AND s.start_time < '2025-09-01'
),
zone_totals AS (
    SELECT
        sport,
        week_start,
        zone_name AS hr_zone,
        SUM(hours_in_zone) AS total_time_in_zone,
        SUM(SUM(hours_in_zone)) OVER (PARTITION BY sport, week_start) AS total_time_sport_week
    FROM intervals
    GROUP BY sport, week_start, zone_name
)
SELECT
    sport,
    week_start,
    hr_zone,
    ROUND(total_time_in_zone, 2) AS total_time_in_zone,
    ROUND((total_time_in_zone / total_time_sport_week) * 100, 2) AS pct_time_in_zone
FROM zone_totals
ORDER BY sport, week_start, hr_zone;
    """
    engine = get_engine()
    df = safe_query(query, engine)
    return df

# Meta page, load indexes
@st.cache_data(show_spinner=True)
def load_indexes_from_db():
    engine = get_engine()
    query = """
        SELECT 
            training_season,
            LOWER(TRIM(index_name)) AS index_name,
            LOWER(TRIM(scope)) AS scope,
            index_value::numeric AS index_value,
            calculated_on,
            source_page
        FROM public.training_indexes
        WHERE index_value IS NOT NULL
        ORDER BY training_season, index_name, calculated_on;
    """
    df = safe_query(query, engine)
    df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")
    return df

# Deload consistency index calculation. Page 2: Volume
def compute_deload_consistency(df_dload):
    """Compute Deload Consistency Index (DCI₂) for each season."""
    DELOAD_DROP_MIN, DELOAD_DROP_MAX, ROLL_WINDOW = 0.6, 0.8, 4

    def calc_deload_metrics(series):
        series = series.fillna(0.0)
        if len(series) < 6:
            return np.nan, 0, 0, 0, []
        roll_mean = series.rolling(window=ROLL_WINDOW, min_periods=1, center=True).mean()
        mask = (series < roll_mean * DELOAD_DROP_MAX) & (series > roll_mean * DELOAD_DROP_MIN)
        idx_weeks = np.where(mask.to_numpy())[0] + 1
        total_weeks = len(series)
        actual = mask.sum()
        expected = max(1, total_weeks // ROLL_WINDOW)
        ratio = actual / expected if expected > 0 else np.nan
        index = 100 * np.exp(-abs(np.log(ratio))) if (ratio and ratio > 0) else 0
        return round(index, 1), int(actual), int(expected), (round(ratio, 2) if np.isfinite(ratio) else np.nan), list(idx_weeks)

    dci_rows = []
    deload_positions = {}
    for season, sub in df_dload.groupby("training_season"):
        weekly_time = sub.sort_values("week_index")["training_time"]
        dci2, actual, expected, ratio, pos = calc_deload_metrics(weekly_time)
        deload_positions[season] = pos
        dci_rows.append({
            "training_season": season,
            "Actual_deloads": actual,
            "Expected_deloads": expected,
            "Ratio": ratio,
            "DCI₂": dci2
        })

    dci_df = pd.DataFrame(dci_rows).sort_values("training_season")
    return dci_df, deload_positions

def compute_svpi(pivot_time: pd.DataFrame, selected_seasons: list[str], growth_threshold: float = 1.05):
    """
    Compute Season Volume Periodisation Index (SVPI).
    Evaluates quarter-to-quarter growth in training time for each season.
    """
    def q_means_from_series(weekly: np.ndarray):
        parts = np.array_split(np.asarray(weekly, dtype=float), 4)
        m = lambda a: float(np.nanmean(a)) if len(a) else np.nan
        return tuple(m(p) for p in parts)

    def safe_ratio(num, den):
        if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
            return np.nan
        return num / den

    def longest_true_run(bools):
        best = cur = 0
        for b in bools:
            cur = cur + 1 if b else 0
            best = max(best, cur)
        return best

    all_seasons = [s for s in pivot_time.columns if s != "2025/26"]
    qmap = {s: q_means_from_series(pivot_time[s].fillna(0.0).to_numpy(float)) for s in all_seasons}

    rows = []
    for s in selected_seasons:
        q1, q2, q3, q4 = qmap.get(s, (np.nan, np.nan, np.nan, np.nan))
        idx = all_seasons.index(s)
        prev_s = all_seasons[idx - 1] if idx - 1 >= 0 else None
        next_s = all_seasons[idx + 1] if idx + 1 < len(all_seasons) else None

        r_prev = safe_ratio(q1, qmap.get(prev_s, (np.nan, np.nan, np.nan, np.nan))[3]) if prev_s else np.nan
        r_12, r_23, r_34 = safe_ratio(q2, q1), safe_ratio(q3, q2), safe_ratio(q4, q3)
        r_next = safe_ratio(qmap.get(next_s, (np.nan, np.nan, np.nan, np.nan))[0], q4) if next_s else np.nan

        ratios = [r_prev, r_12, r_23, r_34, r_next]
        positives = [np.isfinite(r) and r > growth_threshold for r in ratios]
        max_run = longest_true_run(positives)
        total_pos = sum(positives)

        if max_run >= 3: svpi = 100
        elif max_run == 2: svpi = 75
        elif total_pos >= 3: svpi = 70
        elif total_pos == 2: svpi = 60
        elif total_pos == 1: svpi = 50
        else: svpi = 0

        rows.append({
            "training_season": s,
            "Q1_mean": round(q1, 2),
            "Q2_mean": round(q2, 2),
            "Q3_mean": round(q3, 2),
            "Q4_mean": round(q4, 2),
            "SVPI": int(svpi),
        })

    return pd.DataFrame(rows).sort_values("training_season")
    
# Consistency page: upload Deload consistency index for total consistency calc
def load_index_from_db(index_name: str, scope: str = "overall"):
    """Load average index values from training_indexes table by name & scope."""
    engine = get_engine()
    query = text(f"""
        SELECT 
            LOWER(TRIM(scope)) AS scope,
            TRIM(training_season) AS training_season,
            ROUND(AVG(index_value)::numeric, 2) AS "{index_name.upper()}"
        FROM public.training_indexes
        WHERE LOWER(index_name) = LOWER(:index_name)
        AND LOWER(scope) = LOWER(:scope)
        GROUP BY training_season, scope
        ORDER BY training_season;
    """)
    with engine.begin() as conn:
        df = pd.read_sql_query(query, conn, params={"index_name": index_name, "scope": scope})
    return df

# Intensity page: Seve Indexes to DB
def _clean_rows(rows: list[dict]) -> list[dict]:
    """Keep only valid numeric rows and normalize names."""
    cleaned = []
    for r in rows:
        try:
            val = float(r.get("index_value", float("nan")))
        except Exception:
            continue
        if isfinite(val):
            cleaned.append({
                "training_season": str(r["training_season"]),
                "index_name": str(r["index_name"]).strip().lower(),
                "scope": str(r.get("scope", "overall")).strip().lower(),
                "index_value": val,
                "calculated_on": r.get("calculated_on", datetime.now()),
                "source_page": str(r.get("source_page", "HR_Zones_Page")),
            })
    return cleaned


def upsert_training_indexes(engine, rows: list[dict]) -> str:
    """Executemany upsert with retry logic (safe for Neon free tier)."""
    rows = _clean_rows(rows)
    if not rows:
        return "⚠️ Nothing to save (no valid rows)."

    stmt = text("""
        INSERT INTO public.training_indexes
            (training_season, index_name, scope, index_value, calculated_on, source_page)
        VALUES
            (:training_season, :index_name, :scope, :index_value, :calculated_on, :source_page)
        ON CONFLICT (training_season, index_name, scope)
        DO UPDATE SET
            index_value   = EXCLUDED.index_value,
            calculated_on = EXCLUDED.calculated_on,
            source_page   = EXCLUDED.source_page;
    """)

    last_err = None
    for attempt in range(1, 4):  # 3 attempts
        try:
            with engine.begin() as conn:
                conn.execute(stmt, rows)
            return f"✅ Indexes saved to DB."
        except Exception as e:
            last_err = e
            try:
                engine.dispose()
            except Exception:
                pass
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"DB upsert failed after retries: {last_err}")


def build_index_upload_rows(overall_scores, ipi_final) -> list[dict]:
    """Prepare upload rows for Intensity page."""
    upload_rows = []

    # Intensity Balance
    for _, r in overall_scores.iterrows():
        upload_rows.append({
            "training_season": r["training_season"],
            "index_name": "intensity_balance",  # lowercase for META
            "scope": "overall",
            "index_value": float(r["Triathlon_Score"]),
            "calculated_on": datetime.now(),
            "source_page": "HR_Zones_Page",
        })

    # IPI (Intensity Periodisation)
    for _, r in ipi_final.iterrows():
        upload_rows.append({
            "training_season": r["training_season"],
            "index_name": "ipi_intensity",
            "scope": "overall",
            "index_value": float(r["Intensity_Periodisation_Index"]),
            "calculated_on": datetime.now(),
            "source_page": "HR_Zones_Page",
        })

    return upload_rows

# Balance page: save indexes
def save_indexes_to_db(engine, upload_rows, source_page: str = "Unknown"):
    """
    Save calculated index values to the training_indexes table.
    Uses ON CONFLICT for safe upsert.
    """
    insert_query = text("""
        INSERT INTO public.training_indexes 
            (training_season, index_name, scope, index_value, calculated_on, source_page)
        VALUES 
            (:training_season, :index_name, :scope, :index_value, :calculated_on, :source_page)
        ON CONFLICT (training_season, index_name, scope)
        DO UPDATE SET
            index_value   = EXCLUDED.index_value,
            calculated_on = EXCLUDED.calculated_on,
            source_page   = EXCLUDED.source_page;
    """)

    try:
        with engine.begin() as conn:
            for row in upload_rows:
                row["source_page"] = source_page
                row["calculated_on"] = datetime.now()
                conn.execute(insert_query, row)
        st.success("✅ Indexes saved automatically to DB.")
        return True, None
    except Exception as e:
        st.error(f"❌ Failed to save indexes to DB: {e}")
        return False, str(e)
    
# Plan and track page: load zone distribution data
@st.cache_data(show_spinner=True)
def load_weekly_time_in_zones(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Return weekly HR-zone time (hours) between timestamps."""
    engine = get_engine()
    query = """
        SELECT
            DATE_TRUNC('week', s.start_time)::date AS week_start,
            h.sport,
            h.zone_name AS hr_zone,
            ROUND(SUM(h.seconds_in_zone) / 3600.0, 3) AS hours
        FROM hr_zone_summary h
        JOIN session s ON h.file_uid = s.file_uid AND h.sport = s.sport
        WHERE s.sport IN ('cycling', 'running', 'swimming', 'training')
          AND s.start_time >= %(start)s
          AND s.start_time <  %(end)s
        GROUP BY DATE_TRUNC('week', s.start_time), h.sport, h.zone_name
        ORDER BY week_start, h.sport, h.zone_name;
    """
    dfz = pd.read_sql(query, engine, params={"start": start_ts, "end": end_ts})
    return dfz if not dfz.empty else pd.DataFrame(columns=["week_start", "hr_zone", "hours"])

@st.cache_data(show_spinner=True)
def load_season_zone_mix(season_label: str) -> pd.DataFrame:
    """Return total hours and % per HR zone for a given season label."""
    from core.utils import season_dates
    from core.constants import ZONE_ORDER

    start_ts, end_ts = season_dates(season_label)
    dfw = load_weekly_time_in_zones(start_ts, end_ts)
    total = (
        dfw.groupby("hr_zone")["hours"]
        .sum()
        .reindex(ZONE_ORDER)
        .fillna(0)
        .reset_index()
    )
    s = total["hours"].sum()
    total["percent"] = (total["hours"] / s * 100.0) if s > 0 else 0.0
    return total



