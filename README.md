# Garmin Data Studio

**Live Demo:** [garmin-data-studio.streamlit.app](https://garmin-data-studio.streamlit.app)

---

## Overview

Garmin Data Studio turns multi-year Garmin activity data into an interactive analytical dashboard built with **Streamlit** and **PostgreSQL**.

It provides clear insights into training **volume**, **intensity**, **consistency**, and **season planning** — designed for endurance athletes and coaches, with a focus on triathlon.

**Project goals:**
1. Demonstrate data analytics and visualization skills end-to-end.  
2. Serve as a portfolio and learning project in **sports data analytics**.  
3. Derive practical insights to support smarter training decisions.

---

## Purpose

The dashboard shows how raw fitness data can be transformed into useful insights for training optimization.  
It highlights experience in:

- Building ETL data pipelines with **PostgreSQL**  
- Using **Python** and **Pandas** for data processing and analysis  
- Designing modular apps in **Streamlit**  
- Deploying and maintaining production dashboards in the cloud  

---

## Key Features

| Section | Description |
|----------|-------------|
| **Scope** | Season summaries: total hours, sessions, and weekly averages |
| **Volume** | Cumulative and weekly load tracking by sport |
| **Consistency** | Streaks, rest periods, and consistency indexes |
| **Intensity** | HR Zone distribution, 80/15/5 ratio, Intensity Periodisation Index |
| **Sport Balance** | Swim–bike–run ratio with visual trends |
| **Meta Panel** | Season-level summary indexes |
| **Plan & Track Next Season** | KPIs and metrics for future training blocks |

---

## Technical Stack

| Layer | Tools / Technologies |
|-------|----------------------|
| **Frontend** | Streamlit |
| **Backend / Data** | Python, Pandas, NumPy, Matplotlib, SQLAlchemy |
| **Database** | PostgreSQL (NeonDB Cloud) |
| **Environment** | `.env` configuration, schema-based ETL |
| **Deployment** | Streamlit Cloud with GitHub CI/CD |

---

## What This Project Demonstrates

- Real-world data analytics on multi-year endurance training data  
- Integration of **SQL**, **Python**, and **Streamlit** for dynamic insights  
- Modular, production-ready app structure (`core/`, `pages/`)  
- Proven ability to **debug and fix production issues**  
- Complete data workflow: **from processing → visualization → deployment**  
- Practical knowledge of **training load, intensity, and periodisation**

---

## Architecture Overview

Garmin FIT files → Python ETL → PostgreSQL database → Streamlit visualization

All metrics are calculated dynamically via SQL and visualized interactively in Streamlit.

---

## Deployment

- Hosted on **Streamlit Cloud**  
- Connected to **NeonDB PostgreSQL**  
- Automatically redeployed after each GitHub commit to `main`  

---

## Next Steps

- Add Power BI integration for advanced visual reports  
- Build a machine learning module for training load forecasting  

---

## Contact

**Aleh Dudka**  
Kraków, Poland  
[LinkedIn](https://www.linkedin.com/in/olegdudka/) • [GitHub](https://github.com/dudkinov)
