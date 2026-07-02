<div align="center">

# Sales Pulse AI

**AI-powered sales analytics dashboard for e-commerce data**

Built on the Olist Brazilian E-Commerce dataset — combining SQL analytics, Prophet time-series forecasting, and LLM-driven business intelligence in a single, production-style Streamlit application.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-07405E?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Prophet](https://img.shields.io/badge/Forecasting-Prophet-3B82F6?style=flat)](https://facebook.github.io/prophet/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20GPT--OSS%20120B-10B981?style=flat)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## Overview

Sales Pulse AI turns a raw, multi-table e-commerce dataset into a decision-ready analytics product. It ingests and cleans the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), builds a clean revenue master table, and exposes it through an interactive dashboard with:

- **Descriptive analytics** — revenue by category, state, payment method, and time
- **Predictive analytics** — a Prophet forecasting model with honest, out-of-sample accuracy reporting
- **Generative analytics** — an LLM layer (Groq / `openai/gpt-oss-120b`) that turns raw numbers into executive-ready written insights, and translates natural-language questions into safe, validated SQL

The goal isn't just to display charts — it's to demonstrate a realistic, end-to-end data product: ingestion, cleaning, modeling, serving, and AI augmentation.

---

## Features

| Module | Description |
|---|---|
| **KPI Strip** | Total revenue, orders, AOV, unique customers, top state, top category — at a glance |
| **Revenue Breakdown** | Top categories and states ranked by revenue, side-by-side |
| **Product Deep-Dive** | Multi-metric category table (revenue, order volume, avg price, freight-to-price ratio) with AI-generated commentary |
| **Payment Mix** | Revenue and order share by payment method, visualized |
| **Customer Cohorts** | New vs. returning customers per month, based on true person-level identity (not order-level IDs) |
| **Revenue Forecast** | 3-month Prophet forecast with confidence intervals, plus a dedicated out-of-sample holdout evaluation panel (no inflated in-sample metrics) |
| **State Anomaly Detection** | Flags states with >30% month-over-month revenue swings, with one-click AI executive briefings |
| **Natural Language Data Explorer** | Ask questions in plain English; the LLM converts them to SQL, runs the query live against the database, and returns results in-app |

---

## AI Layer

This project intentionally separates what the data says from what an LLM says about it:

- **NL to SQL**: User questions are converted to SQLite `SELECT` statements by an LLM, then passed through an allowlist safety filter before execution — only single `SELECT` statements are permitted; any `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ATTACH`, `PRAGMA`, etc. is rejected outright.
- **Business Insight Generation**: Detected anomalies and category performance tables are summarized into structured, numbers-grounded executive briefings — not generic LLM filler.
- **Model**: [`openai/gpt-oss-120b`](https://groq.com/) served via the Groq API, chosen for low-latency inference suitable for an interactive dashboard.
- **Resilience**: All LLM calls go through a centralized helper with exponential-backoff retry on transient API failures.

---

## Architecture

```
sales-pulse-ai/
│
├── data/                          # Raw Olist CSVs + generated SQLite DB
├── notebooks/
│   └── 01_data_exploration.ipynb  # ETL pipeline: clean, merge, load to SQL
├── src/
│   └── data_utils.py              # All business logic: queries, forecasting, LLM calls
├── app.py                         # Streamlit presentation layer (no business logic)
├── requirements.txt
└── .env                           # GROQ_API_KEY (not committed)
```

**Design principles:**
- **Separation of concerns** — `app.py` is pure presentation; every query, model fit, and LLM call lives in `data_utils.py`.
- **Centralized DB access** — all reads go through a single `run_query()` function for consistent logging and error handling.
- **Defensive data engineering** — null-handling decisions are explicit and justified (drop vs. fill), not silent.
- **Honest forecasting** — accuracy is reported on a true train/holdout split, not in-sample fit.

### Data Pipeline

```
Raw CSVs (customers, orders, items, products, payments)
        |
        v
Filter to delivered orders only
        |
        v
Clean nulls, aggregate payments, merge to order-item grain
        |
        v
sales_master table  ->  SQLite (sales_pulse.db)
        |
        v
Streamlit app reads via SQLAlchemy
```

> **Note on identity resolution:** Olist generates a new `customer_id` per order, which would make every purchase look like a "new" customer. This pipeline uses `customer_unique_id` (the true person-level identifier) for cohort and repeat-customer analysis — a common real-world data trap that's handled explicitly here rather than glossed over.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)
- The [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CSV files)

### 1. Clone the repository

```bash
git clone https://github.com/omshewalegit/sales-pulse-ai.git
cd sales-pulse-ai
```

### 2. Set up a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Place the dataset

Download the Olist dataset from Kaggle and place the CSVs in the `data/` folder:

```
data/
├── olist_customers_dataset.csv
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_products_dataset.csv
└── ...
```

### 6. Run the ETL pipeline

Open and run all cells in `notebooks/01_data_exploration.ipynb`. This cleans the raw data and builds `data/sales_pulse.db`.

### 7. Launch the dashboard

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / App | Streamlit |
| Data Processing | Pandas |
| Database | SQLite + SQLAlchemy |
| Forecasting | Prophet (Meta) |
| LLM / AI Insights | Groq API — `openai/gpt-oss-120b` |
| Visualization | Matplotlib |
| Evaluation | scikit-learn (MAPE) |

---

## Forecasting Methodology

The revenue forecast uses Facebook Prophet with:
- Weekly/daily seasonality disabled (input is monthly-aggregated)
- Yearly seasonality only enabled with 24+ months of history (insufficient data otherwise produces spurious seasonal patterns)
- A dedicated holdout evaluation — the model is trained on all but the last 3 months and scored against actuals it never saw, surfacing a realistic MAPE instead of an inflated in-sample number

This distinction matters: an in-sample fit will always look better than a model performs in production. This project reports the honest number.

---

## Security Notes

- All LLM-generated SQL passes through an allowlist filter (`is_safe_query`) before execution — only `SELECT` statements are permitted.
- `.env` and `venv/` are git-ignored — no credentials are committed.
- In a real production deployment, the LLM-facing DB connection should additionally use a read-only database user as defense-in-depth beyond the application-level check.

---

## Roadmap

- [ ] Migrate raw data storage to Git LFS or external object storage (S3 / GCS)
- [ ] Add a `requirements-dev.txt` with linting/testing tools
- [ ] Containerize with Docker for one-command setup
- [ ] Add unit tests for `data_utils.py` query functions
- [ ] Deploy a live demo (Streamlit Community Cloud)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Olist](https://olist.com/) for the public Brazilian e-commerce dataset
- [Meta Prophet](https://facebook.github.io/prophet/) for the forecasting library
- [Groq](https://groq.com/) for fast LLM inference

---

<div align="center">

Built by [Om Shewale](https://github.com/omshewalegit)

</div>
