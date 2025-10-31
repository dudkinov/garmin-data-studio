import streamlit as st

st.set_page_config(layout="wide", page_title="Garmin Data Studio", page_icon="📊")

# ---------- Header ----------
st.title("📊 Garmin Data Studio")
st.caption("A personal analytics platform that turns multi-year Garmin data into clear, season-level insights for planning, tracking, and continuous improvement.")
st.divider()

# ---------- Business Case + Dataset ----------
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("💼 Business Case")
    st.markdown(
        "- **Problem:** Most fitness platforms, such as **Garmin** and **Strava**, focus on daily stats, missing the big-picture perspective.\n"
        "- **Solution:** This project delivers **season-level** analytics and insights.\n"
        "- **Value:** Athletes and coaches can optimize long-term progress with data-driven decisions."
    )

with col_right:
    st.subheader("📂 Dataset")
    st.markdown(
        "- **Source:** Garmin Connect (personal training data exported as FIT files).\n"
        "- **Scope:** 5 years of triathlon training (cycling, running, swimming, strength).\n"
        "- **Content:** Activity/session/lap data with metrics like power, pace, heart rate, cadence, distance, etc."
    )

st.divider()

# ---------- Tech Stack + GitHub ----------
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("🛠️ Tech Stack")
    st.markdown(
        "- **Database (Cloud, live-updated):** PostgreSQL, SQLAlchemy\n"
        "- **ETL & Data Cleaning:** Python (Pandas, Numpy, FitDecode)\n"
        "- **Visualization:** Streamlit, Matplotlib\n"
        "- **Infrastructure:** Neon (DB hosting)"
    )

with col_right:
    st.subheader("🧷 GitHub Repository")
    st.write("Source code for ETL (FIT → PostgreSQL), data model, and Streamlit dashboards.")
    st.link_button("Open on GitHub ↗", "https://github.com/dudkinov/garmin-streamlit-analytics")

st.divider()

# ---------- Data Flow ----------
st.subheader("🔗 Data Flow")
st.markdown(
    "Garmin Connect app → Garmin FIT files → ETL pipeline → PostgreSQL → Streamlit → Analytics"
)

st.divider()

# ---------- Quick links to analysis ----------
st.subheader("🚀 Jump into the Analysis")

def nav_button(label: str, description: str, page: str, key: str):
    st.markdown(f"**{label}**  \n{description}")
    # Always use a real button so they all render the same
    if st.button("Open", key=key, use_container_width=True):
        st.switch_page(page)

r1c1, r1c2, r1c3, r1c4 = st.columns(4)
with r1c1:
    nav_button("Analysis Scope", "Business context & goals", "pages/1_1_Scope.py", "btn_scope")
with r1c2:
    nav_button("Volume", "Weekly hours, trends, totals", "pages/2_2_Volume.py", "btn_volume")
with r1c3:
    nav_button("Consistency", "Frequency & gaps (ACI)", "pages/3_3_Consistency.py", "btn_consistency")
with r1c4:
    nav_button("Intensity", "Time in HR zones", "pages/4_4_Intensity.py", "btn_intensity")

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
with r2c1:
    nav_button("Sport Balance", "Discipline distribution", "pages/5_5_Sport_Balance.py", "btn_balance")
with r2c2:
    nav_button("META Panel", "Composite season indexes", "pages/6_6_Meta_Pannel.py", "btn_meta")
with r2c3:
    nav_button("Plan & Track", "Plan and execution", "pages/7_7_Plan_&_Track_Next_Season.py", "btn_planning")

st.divider()
st.caption("This landing page presents the business value, dataset, and architecture behind the project. Use the links above to dive into the analysis.")
