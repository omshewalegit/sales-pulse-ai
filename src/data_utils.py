# """
# data_utils.py
# Core data access and AI utility functions for Sales Pulse AI.
# Handles DB connections, SQL analytics queries, forecasting, and LLM-powered insights.
# """

# import os
# import logging
# from functools import lru_cache
# from typing import Optional

# import pandas as pd
# from sqlalchemy import create_engine, text
# from sqlalchemy.engine import Engine
# from dotenv import load_dotenv
# from groq import Groq, GroqError
# from prophet import Prophet

# # ---------------------------------------------------------------------------
# # Setup
# # ---------------------------------------------------------------------------

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
# )
# logger = logging.getLogger("data_utils")

# DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sales_pulse.db")
# GROQ_MODEL = "llama-3.3-70b-versatile"

# FORBIDDEN_SQL_KEYWORDS = (
#     "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
#     "CREATE", "TRUNCATE", "ATTACH", "PRAGMA", "REPLACE"
# )


# # ---------------------------------------------------------------------------
# # Custom exceptions
# # ---------------------------------------------------------------------------

# class UnsafeQueryError(Exception):
#     """Raised when a generated SQL query fails the safety check."""


# class InsightGenerationError(Exception):
#     """Raised when the LLM fails to generate a usable insight."""


# # ---------------------------------------------------------------------------
# # DB connection
# # ---------------------------------------------------------------------------

# @lru_cache(maxsize=1)
# def get_engine() -> Engine:
#     """
#     Returns a cached SQLAlchemy engine pointing to the local SQLite DB.
#     Cached so we don't reopen a new connection pool on every Streamlit rerun.
#     """
#     if not os.path.exists(DB_PATH):
#         raise FileNotFoundError(
#             f"Database not found at {DB_PATH}. Run the data pipeline notebook first."
#         )
#     logger.info("Creating DB engine at %s", DB_PATH)
#     return create_engine(f"sqlite:///{DB_PATH}")


# def run_query(engine: Engine, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
#     """
#     Centralized query runner with error handling and logging.
#     Always use this instead of pd.read_sql directly so failures are caught consistently.
#     """
#     try:
#         with engine.connect() as conn:
#             return pd.read_sql(text(sql), conn, params=params)
#     except Exception as e:
#         logger.error("Query failed: %s | SQL: %s", e, sql)
#         raise


# # ---------------------------------------------------------------------------
# # Analytics queries
# # ---------------------------------------------------------------------------

# def get_top_categories(engine: Engine, limit: int = 10) -> pd.DataFrame:
#     sql = """
#         SELECT product_category_name,
#                ROUND(SUM(price), 2) AS total_revenue,
#                COUNT(DISTINCT order_id) AS total_orders
#         FROM sales_master
#         GROUP BY product_category_name
#         ORDER BY total_revenue DESC
#         LIMIT :limit
#     """
#     return run_query(engine, sql, {"limit": limit})


# def get_state_revenue(engine: Engine, limit: int = 10) -> pd.DataFrame:
#     sql = """
#         SELECT customer_state,
#                ROUND(SUM(price), 2) AS total_revenue,
#                COUNT(DISTINCT order_id) AS total_orders
#         FROM sales_master
#         GROUP BY customer_state
#         ORDER BY total_revenue DESC
#         LIMIT :limit
#     """
#     return run_query(engine, sql, {"limit": limit})


# def get_monthly_trend(
#     engine: Engine,
#     start_date: str = "2017-01-01",
#     end_date: str = "2018-08-01",
# ) -> pd.DataFrame:
#     """
#     Returns monthly revenue aggregated for forecasting.
#     end_date is exclusive — by default the last (incomplete) month is excluded,
#     since partial months distort trend/forecast quality.
#     """
#     sql = """
#         SELECT strftime('%Y-%m-01', order_purchase_timestamp) AS ds,
#                SUM(price) AS y
#         FROM sales_master
#         WHERE order_purchase_timestamp >= :start_date
#           AND order_purchase_timestamp < :end_date
#         GROUP BY ds
#         ORDER BY ds
#     """
#     df = run_query(engine, sql, {"start_date": start_date, "end_date": end_date})
#     if df.empty:
#         raise ValueError("No data returned for monthly trend — check date range.")
#     df["ds"] = pd.to_datetime(df["ds"])
#     return df


# def get_significant_changes(engine: Engine, threshold: float = 30.0) -> tuple[pd.DataFrame, str]:
#     """
#     Detects month-over-month revenue swings per state above `threshold`%.
#     Excludes the most recent calendar month if it appears incomplete
#     (heuristic: fewer than 15 distinct order days recorded for it).
#     Returns (changes_df, month_analyzed).
#     """
#     sql = """
#         SELECT strftime('%Y-%m', order_purchase_timestamp) AS month,
#                customer_state,
#                ROUND(SUM(price), 2) AS revenue
#         FROM sales_master
#         GROUP BY month, customer_state
#         ORDER BY month
#     """
#     df = run_query(engine, sql)
#     if df.empty:
#         raise ValueError("No data available to detect significant changes.")

#     # Heuristic: drop the latest month if it looks incomplete
#     completeness_sql = """
#         SELECT strftime('%Y-%m', order_purchase_timestamp) AS month,
#                COUNT(DISTINCT DATE(order_purchase_timestamp)) AS days_with_orders
#         FROM sales_master
#         GROUP BY month
#         ORDER BY month DESC
#         LIMIT 1
#     """
#     latest_check = run_query(engine, completeness_sql)
#     if not latest_check.empty and latest_check.iloc[0]["days_with_orders"] < 15:
#         incomplete_month = latest_check.iloc[0]["month"]
#         logger.info("Excluding likely-incomplete month: %s", incomplete_month)
#         df = df[df["month"] != incomplete_month]

#     df["prev_revenue"] = df.groupby("customer_state")["revenue"].shift(1)
#     df["pct_change"] = ((df["revenue"] - df["prev_revenue"]) / df["prev_revenue"]) * 100

#     latest_month = df["month"].max()
#     changes = df[
#         (df["month"] == latest_month) & (df["pct_change"].abs() > threshold)
#     ].dropna(subset=["pct_change"])

#     return changes, latest_month


# # ---------------------------------------------------------------------------
# # Forecasting
# # ---------------------------------------------------------------------------

# def get_forecast(monthly_data: pd.DataFrame, periods: int = 3) -> pd.DataFrame:
#     """
#     Fits a Prophet model on monthly aggregated data and returns the forecast frame.
#     Disables daily/weekly seasonality since the input is already monthly-aggregated —
#     fitting sub-monthly seasonality on monthly data causes spurious overfitting.
#     """
#     if len(monthly_data) < 6:
#         raise ValueError(
#             f"Only {len(monthly_data)} data points available; "
#             "need at least 6 months for a minimally stable forecast."
#         )

#     model = Prophet(
#         weekly_seasonality=False,
#         daily_seasonality=False,
#         yearly_seasonality=len(monthly_data) >= 24,  # only enable if 2+ years of data
#     )
#     model.fit(monthly_data)

#     future = model.make_future_dataframe(periods=periods, freq="MS")
#     forecast = model.predict(future)
#     return forecast


# def evaluate_forecast_holdout(monthly_data: pd.DataFrame, holdout_periods: int = 3) -> dict:
#     """
#     Honest out-of-sample evaluation: trains on all but the last `holdout_periods` months,
#     predicts those held-out months, and compares against actuals.
#     Returns a dict with MAPE and the comparison table — used to surface real model
#     performance rather than in-sample (overfit) accuracy.
#     """
#     from sklearn.metrics import mean_absolute_percentage_error

#     if len(monthly_data) <= holdout_periods + 3:
#         raise ValueError("Not enough data to create a meaningful train/holdout split.")

#     train = monthly_data.iloc[:-holdout_periods]
#     test = monthly_data.iloc[-holdout_periods:]

#     model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
#     model.fit(train)

#     future = model.make_future_dataframe(periods=holdout_periods, freq="MS")
#     forecast = model.predict(future)

#     comparison = test.merge(forecast[["ds", "yhat"]], on="ds")
#     mape = mean_absolute_percentage_error(comparison["y"], comparison["yhat"])

#     return {
#         "mape": mape,
#         "comparison": comparison,
#         "n_train_months": len(train),
#         "n_test_months": len(test),
#     }


# # ---------------------------------------------------------------------------
# # LLM: schema info, NL→SQL, business insights
# # ---------------------------------------------------------------------------

# @lru_cache(maxsize=1)
# def get_groq_client() -> Groq:
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise EnvironmentError(
#             "GROQ_API_KEY not found. Check your .env file is in the project root."
#         )
#     return Groq(api_key=api_key)


# def get_schema_info() -> str:
#     return """
# Table name: sales_master
# Columns:
# - order_id (text)
# - customer_id (text)
# - order_status (text) - typically 'delivered' (table is pre-filtered to delivered orders)
# - order_purchase_timestamp (datetime, format 'YYYY-MM-DD HH:MM:SS')
# - price (float) - price of an individual line item
# - freight_value (float) - shipping cost for that line item
# - product_category_name (text) - e.g. 'beleza_saude', 'informatica_acessorios'
# - customer_city (text)
# - customer_state (text) - two-letter Brazilian state code, e.g. 'SP', 'RJ', 'MG'
# - payment_type (text) - e.g. 'credit_card', 'boleto', 'voucher'
# - payment_value (float)
# """


# def is_safe_query(sql: str) -> bool:
#     """
#     Basic allowlist-style safety check before executing any LLM-generated SQL.
#     Only single SELECT statements are permitted; anything else is rejected.
#     This is a defense-in-depth measure, not a substitute for running with a
#     read-only DB user in a real production deployment.
#     """
#     if not sql or not sql.strip():
#         return False

#     cleaned = sql.strip().rstrip(";")
#     if ";" in cleaned:
#         return False  # block stacked queries

#     upper = cleaned.upper()
#     if not upper.startswith("SELECT"):
#         return False

#     return not any(keyword in upper for keyword in FORBIDDEN_SQL_KEYWORDS)


# def natural_language_to_sql(question: str, client: Groq) -> str:
#     """
#     Converts a natural-language question into a SQLite SQL query using the LLM.
#     Returns the literal string 'NOT_RELEVANT' if the question is unrelated
#     to the sales dataset, so the caller can short-circuit before hitting the DB.
#     """
#     schema = get_schema_info()
#     prompt = f"""You are a SQL expert. Given this SQLite table schema:
# {schema}

# Convert this question into a single valid SQLite SQL query: "{question}"

# Rules:
# - If the question is NOT related to e-commerce sales data (general knowledge, current events,
#   sports, anything outside this schema), respond with exactly: NOT_RELEVANT
# - Return ONLY the SQL query, no explanation, no markdown, no backticks
# - Use proper SQLite syntax
# - Always include the aggregated/calculated column (COUNT, SUM, AVG, etc.) in the SELECT
#   clause itself, not only in ORDER BY
# - Always add LIMIT 20 unless the question asks for a single aggregated value
# - Only SELECT statements are allowed — never INSERT/UPDATE/DELETE/DROP/ALTER
# """
#     try:
#         response = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#             timeout=20,
#         )
#     except GroqError as e:
#         logger.error("Groq API call failed: %s", e)
#         raise InsightGenerationError(f"LLM request failed: {e}") from e

#     sql = response.choices[0].message.content.strip()
#     sql = sql.replace("```sql", "").replace("```", "").strip()
#     return sql


# def ask_data_question(question: str, engine: Engine, client: Groq) -> dict:
#     """
#     Full pipeline for the 'Ask Your Data' feature:
#     NL question -> SQL -> safety check -> execution -> structured result.
#     Centralizing this here keeps app.py free of business logic.
#     Returns a dict: {status, sql, data (DataFrame or None), message}
#     """
#     sql = natural_language_to_sql(question, client)

#     if sql == "NOT_RELEVANT":
#         return {"status": "not_relevant", "sql": None, "data": None,
#                 "message": "This question doesn't relate to the sales dataset."}

#     if not is_safe_query(sql):
#         logger.warning("Blocked unsafe generated query: %s", sql)
#         return {"status": "unsafe", "sql": sql, "data": None,
#                 "message": "Generated query failed the safety check."}

#     try:
#         result = run_query(engine, sql)
#     except Exception as e:
#         return {"status": "error", "sql": sql, "data": None,
#                 "message": f"Query execution failed: {e}"}

#     return {"status": "ok", "sql": sql, "data": result, "message": None}


# def generate_ai_insight(changes_df: pd.DataFrame, month: str, client: Groq) -> str:
#     """
#     Sends detected month-over-month anomalies to the LLM and returns a
#     stakeholder-ready business insight summary in plain English.
#     """
#     if changes_df.empty:
#         return "No significant month-over-month changes were detected for this period."

#     changes_text = changes_df[["customer_state", "revenue", "pct_change"]].to_string(index=False)

#     confidence_note = ""
#     if len(changes_df) < 3:
#         confidence_note = (
#             "\nNote: only a small number of states show significant change this month — "
#             "treat conclusions as directional, not definitive."
#         )

#     prompt = f"""You are a business data analyst. Below is data showing month-over-month
# revenue changes (%) by state for an e-commerce company, for the month of {month}.

# {changes_text}
# {confidence_note}

# Write a concise business insight summary (4-5 bullet points) highlighting:
# 1. The most significant growth states and possible business implications
# 2. Any concerning declines
# 3. One actionable recommendation for the business

# Keep it professional and data-driven, suitable for a business stakeholder report."""

#     try:
#         response = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             timeout=20,
#         )
#     except GroqError as e:
#         logger.error("Groq API call failed: %s", e)
#         raise InsightGenerationError(f"LLM request failed: {e}") from e

#     return response.choices[0].message.content
"""
data_utils.py
=============
Core data-access and AI utility layer for Sales Pulse AI.

Responsibilities
----------------
- SQLite connection management (cached engine, centralised query runner)
- Analytics queries: KPIs, categories, states, payment mix, cohorts,
  delivery metrics, review sentiment, revenue heatmap
- Time-series forecasting with Prophet (fit + holdout evaluation)
- LLM integration: NL→SQL translation and AI business-insight generation
  via Groq (Llama 3.3 70B), with exponential-backoff retry

Design principles
-----------------
- Every public function is fully type-annotated and has a NumPy-style docstring.
- Database I/O is centralised through `run_query` so failures are caught,
  logged, and surfaced consistently.
- The Groq client and SQLAlchemy engine are each created once and cached;
  Streamlit reruns do not create duplicate connections.
- No business logic lives in app.py — only presentation.

Author : Sales Pulse AI
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, GroqError
from prophet import Prophet
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("data_utils")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH: str = os.path.join(os.path.dirname(__file__), "..", "data", "sales_pulse.db")
GROQ_MODEL: str = "openai/gpt-oss-120b"

# SQL keywords that must never appear in LLM-generated queries.
FORBIDDEN_SQL_KEYWORDS: tuple[str, ...] = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "TRUNCATE", "ATTACH", "PRAGMA", "REPLACE",
)

# Groq retry settings
_GROQ_MAX_RETRIES: int = 3
_GROQ_BACKOFF_BASE: float = 1.5   # seconds; wait = base ** attempt


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class UnsafeQueryError(Exception):
    """Raised when a generated SQL query fails the allowlist safety check."""


class InsightGenerationError(Exception):
    """Raised when the LLM fails to produce a usable response."""


class DataValidationError(Exception):
    """Raised when a DataFrame is missing expected columns or is empty."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    context: str = "",
) -> None:
    """
    Assert that *df* is non-empty and contains every column in *required_columns*.

    Parameters
    ----------
    df : pd.DataFrame
        The frame to validate.
    required_columns : list[str]
        Column names that must be present.
    context : str, optional
        Human-readable description used in the error message.

    Raises
    ------
    DataValidationError
        If the frame is empty or any required column is absent.
    """
    if df.empty:
        raise DataValidationError(
            f"{'[' + context + '] ' if context else ''}DataFrame is unexpectedly empty."
        )
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{'[' + context + '] ' if context else ''}Missing columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Return a cached SQLAlchemy engine pointing at the local SQLite database.

    The engine is created once per process; Streamlit reruns reuse the same
    connection pool instead of opening a new file handle every time.

    Returns
    -------
    Engine
        Connected SQLAlchemy engine.

    Raises
    ------
    FileNotFoundError
        If the SQLite database file does not exist at the expected path.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at '{DB_PATH}'. "
            "Run the data-pipeline notebook (01_data_exploration.ipynb) first."
        )
    logger.info("Initialising DB engine — %s", DB_PATH)
    return create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def run_query(
    engine: Engine,
    sql: str,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Execute a parameterised SQL query and return the result as a DataFrame.

    All database reads in this module go through this function so that
    connection handling, logging, and exception propagation are consistent.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine returned by :func:`get_engine`.
    sql : str
        SQL statement with optional ``:name`` bind parameters.
    params : dict, optional
        Bind-parameter values keyed by name.

    Returns
    -------
    pd.DataFrame
        Query result; may be empty if the query returns no rows.

    Raises
    ------
    Exception
        Re-raises any SQLAlchemy / SQLite exception after logging it.
    """
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception as exc:
        logger.error("Query failed — %s | SQL: %.300s", exc, sql)
        raise


# ---------------------------------------------------------------------------
# Analytics queries
# ---------------------------------------------------------------------------

def get_kpi_summary(engine: Engine) -> dict:
    """
    Compute top-level KPI metrics for the summary strip on the dashboard.

    Returns
    -------
    dict
        Keys: ``total_revenue``, ``total_orders``, ``avg_order_value``,
        ``total_customers``, ``top_state``, ``top_category``.
    """
    sql = """
        SELECT
            ROUND(SUM(price), 2)                    AS total_revenue,
            COUNT(DISTINCT order_id)                AS total_orders,
            ROUND(AVG(price), 2)                    AS avg_order_value,
            COUNT(DISTINCT customer_unique_id)             AS total_customers
        FROM sales_master
    """
    base = run_query(engine, sql)

    top_state_df = run_query(
        engine,
        "SELECT customer_state FROM sales_master "
        "GROUP BY customer_state ORDER BY SUM(price) DESC LIMIT 1",
    )
    top_cat_df = run_query(
        engine,
        "SELECT product_category_name FROM sales_master "
        "GROUP BY product_category_name ORDER BY SUM(price) DESC LIMIT 1",
    )

    row = base.iloc[0]
    return {
        "total_revenue":    float(row["total_revenue"] or 0),
        "total_orders":     int(row["total_orders"] or 0),
        "avg_order_value":  float(row["avg_order_value"] or 0),
        "total_customers":  int(row["total_customers"] or 0),
        "top_state":        top_state_df.iloc[0, 0] if not top_state_df.empty else "N/A",
        "top_category":     top_cat_df.iloc[0, 0] if not top_cat_df.empty else "N/A",
    }


def get_top_categories(engine: Engine, limit: int = 10) -> pd.DataFrame:
    """
    Return the top product categories ranked by total revenue.

    Parameters
    ----------
    engine : Engine
    limit : int, default 10
        Number of categories to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``product_category_name``, ``total_revenue``, ``total_orders``,
        ``avg_item_price``.
    """
    sql = """
        SELECT
            product_category_name,
            ROUND(SUM(price), 2)            AS total_revenue,
            COUNT(DISTINCT order_id)        AS total_orders,
            ROUND(AVG(price), 2)            AS avg_item_price
        FROM sales_master
        GROUP BY product_category_name
        ORDER BY total_revenue DESC
        LIMIT :limit
    """
    df = run_query(engine, sql, {"limit": limit})
    validate_dataframe(df, ["product_category_name", "total_revenue"], "get_top_categories")
    return df


def get_state_revenue(engine: Engine, limit: int = 10) -> pd.DataFrame:
    """
    Return the top Brazilian states ranked by total revenue.

    Parameters
    ----------
    engine : Engine
    limit : int, default 10

    Returns
    -------
    pd.DataFrame
        Columns: ``customer_state``, ``total_revenue``, ``total_orders``,
        ``avg_order_value``.
    """
    sql = """
        SELECT
            customer_state,
            ROUND(SUM(price), 2)            AS total_revenue,
            COUNT(DISTINCT order_id)        AS total_orders,
            ROUND(AVG(price), 2)            AS avg_order_value
        FROM sales_master
        GROUP BY customer_state
        ORDER BY total_revenue DESC
        LIMIT :limit
    """
    df = run_query(engine, sql, {"limit": limit})
    validate_dataframe(df, ["customer_state", "total_revenue"], "get_state_revenue")
    return df


def get_monthly_trend(
    engine: Engine,
    start_date: str = "2017-01-01",
    end_date: str = "2018-08-01",
) -> pd.DataFrame:
    """
    Aggregate revenue by calendar month for time-series forecasting.

    The *end_date* is exclusive; by default the last (likely incomplete) month
    in the dataset is excluded to avoid distorting the trend line.

    Parameters
    ----------
    engine : Engine
    start_date : str, default ``"2017-01-01"``
        Inclusive lower bound (ISO 8601 date string).
    end_date : str, default ``"2018-08-01"``
        Exclusive upper bound.

    Returns
    -------
    pd.DataFrame
        Columns: ``ds`` (datetime), ``y`` (float revenue).

    Raises
    ------
    ValueError
        If no rows are returned for the specified date range.
    """
    sql = """
        SELECT
            strftime('%Y-%m-01', order_purchase_timestamp) AS ds,
            ROUND(SUM(price), 2)                           AS y
        FROM sales_master
        WHERE order_purchase_timestamp >= :start_date
          AND order_purchase_timestamp <  :end_date
        GROUP BY ds
        ORDER BY ds
    """
    df = run_query(engine, sql, {"start_date": start_date, "end_date": end_date})
    if df.empty:
        raise ValueError(
            f"No monthly revenue data found between {start_date} and {end_date}. "
            "Verify the date range against your dataset."
        )
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def get_payment_breakdown(engine: Engine) -> pd.DataFrame:
    """
    Return revenue and order count split by payment method.

    Returns
    -------
    pd.DataFrame
        Columns: ``payment_type``, ``total_revenue``, ``total_orders``,
        ``revenue_share_pct``.
    """
    sql = """
        SELECT
            payment_type,
            ROUND(SUM(price), 2)                                        AS total_revenue,
            COUNT(DISTINCT order_id)                                    AS total_orders,
            ROUND(SUM(price) * 100.0 / SUM(SUM(price)) OVER (), 1)    AS revenue_share_pct
        FROM sales_master
        GROUP BY payment_type
        ORDER BY total_revenue DESC
    """
    df = run_query(engine, sql)
    validate_dataframe(df, ["payment_type", "total_revenue"], "get_payment_breakdown")
    return df


def get_revenue_heatmap_data(engine: Engine) -> pd.DataFrame:
    """
    Build a state × month revenue matrix suitable for a heatmap visualisation.

    Returns
    -------
    pd.DataFrame
        Pivot table: index = ``customer_state``, columns = month strings
        (``YYYY-MM``), values = total revenue (float).
    """
    sql = """
        SELECT
            customer_state,
            strftime('%Y-%m', order_purchase_timestamp) AS month,
            ROUND(SUM(price), 2)                        AS revenue
        FROM sales_master
        GROUP BY customer_state, month
        ORDER BY month
    """
    df = run_query(engine, sql)
    validate_dataframe(df, ["customer_state", "month", "revenue"], "get_revenue_heatmap_data")
    pivot = df.pivot_table(
        index="customer_state",
        columns="month",
        values="revenue",
        aggfunc="sum",
        fill_value=0,
    )
    # Keep only states with meaningful activity (total revenue > 0)
    pivot = pivot.loc[pivot.sum(axis=1) > 0]
    return pivot


def get_product_performance(engine: Engine, limit: int = 15) -> pd.DataFrame:
    """
    Return a multi-metric product-category performance table.

    Combines revenue, order volume, average item price, and freight ratio
    to give a richer view than revenue alone.

    Parameters
    ----------
    engine : Engine
    limit : int, default 15

    Returns
    -------
    pd.DataFrame
        Columns: ``product_category_name``, ``total_revenue``, ``total_orders``,
        ``avg_item_price``, ``avg_freight``, ``freight_ratio_pct``.
    """
    sql = """
        SELECT
            product_category_name,
            ROUND(SUM(price), 2)                                    AS total_revenue,
            COUNT(DISTINCT order_id)                                AS total_orders,
            ROUND(AVG(price), 2)                                    AS avg_item_price,
            ROUND(AVG(freight_value), 2)                            AS avg_freight,
            ROUND(AVG(freight_value) * 100.0 / NULLIF(AVG(price), 0), 1)
                                                                    AS freight_ratio_pct
        FROM sales_master
        GROUP BY product_category_name
        ORDER BY total_revenue DESC
        LIMIT :limit
    """
    df = run_query(engine, sql, {"limit": limit})
    validate_dataframe(df, ["product_category_name", "total_revenue"], "get_product_performance")
    return df


def get_delivery_metrics(engine: Engine) -> dict:
    """
    Compute order fulfilment KPIs: average delivery time and on-time rate.

    Uses ``order_purchase_timestamp`` and ``order_delivered_customer_date``
    from the underlying ``sales_master`` view.  Returns safe defaults when
    either column is absent (older DB schemas may not include them).

    Returns
    -------
    dict
        Keys: ``avg_delivery_days`` (float), ``on_time_rate_pct`` (float),
        ``total_delivered`` (int).
    """
    # Gracefully skip if the DB schema doesn't have delivery columns
    check_sql = "SELECT * FROM sales_master LIMIT 1"
    sample = run_query(engine, check_sql)
    if "order_delivered_customer_date" not in sample.columns:
        logger.warning(
            "order_delivered_customer_date column not found — delivery metrics unavailable."
        )
        return {
            "avg_delivery_days": None,
            "on_time_rate_pct": None,
            "total_delivered": None,
        }

    sql = """
        SELECT
            ROUND(AVG(
                JULIANDAY(order_delivered_customer_date) -
                JULIANDAY(order_purchase_timestamp)
            ), 1) AS avg_delivery_days,
            ROUND(
                SUM(CASE
                    WHEN order_delivered_customer_date <= order_estimated_delivery_date
                    THEN 1 ELSE 0
                END) * 100.0 / COUNT(*), 1
            ) AS on_time_rate_pct,
            COUNT(*) AS total_delivered
        FROM sales_master
        WHERE order_delivered_customer_date IS NOT NULL
    """
    row = run_query(engine, sql).iloc[0]
    return {
        "avg_delivery_days": float(row["avg_delivery_days"] or 0),
        "on_time_rate_pct":  float(row["on_time_rate_pct"] or 0),
        "total_delivered":   int(row["total_delivered"] or 0),
    }


def get_review_score_distribution(engine: Engine) -> pd.DataFrame:
    """
    Return the distribution of customer review scores (1–5).

    Returns
    -------
    pd.DataFrame
        Columns: ``review_score``, ``count``, ``share_pct``.
        Returns an empty DataFrame if ``review_score`` is not in the schema.
    """
    sample = run_query(engine, "SELECT * FROM sales_master LIMIT 1")
    if "review_score" not in sample.columns:
        logger.warning("review_score column not found — skipping review distribution.")
        return pd.DataFrame()

    sql = """
        SELECT
            review_score,
            COUNT(*)                                        AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS share_pct
        FROM sales_master
        WHERE review_score IS NOT NULL
        GROUP BY review_score
        ORDER BY review_score
    """
    return run_query(engine, sql)


def get_customer_cohort_data(engine: Engine) -> pd.DataFrame:
    """
    Classify customers as new vs. returning by month.

    Uses customer_unique_id (the person-level identifier) rather than
    customer_id, since customer_id is generated per-order in this dataset
    and would make every customer look "new" every time.

    A customer is classified as *returning* in month M if they placed an
    order in any month prior to M.

    Returns
    -------
    pd.DataFrame
        Columns: ``month``, ``new_customers``, ``returning_customers``.
    """
    sql = """
        WITH first_order AS (
            SELECT
                customer_unique_id,
                MIN(strftime('%Y-%m', order_purchase_timestamp)) AS first_month
            FROM sales_master
            GROUP BY customer_unique_id
        ),
        monthly AS (
            SELECT
                strftime('%Y-%m', s.order_purchase_timestamp) AS month,
                s.customer_unique_id,
                f.first_month
            FROM sales_master s
            JOIN first_order f USING (customer_unique_id)
        )
        SELECT
            month,
            SUM(CASE WHEN month = first_month THEN 1 ELSE 0 END) AS new_customers,
            SUM(CASE WHEN month > first_month THEN 1 ELSE 0 END)  AS returning_customers
        FROM monthly
        GROUP BY month
        ORDER BY month
    """
    df = run_query(engine, sql)
    return df


def get_significant_changes(
    engine: Engine,
    threshold: float = 30.0,
) -> tuple[pd.DataFrame, str]:
    """
    Detect states with month-over-month revenue swings above *threshold* percent.

    An incomplete latest month (fewer than 15 distinct order days) is
    automatically excluded to avoid noise from partial-period data.

    Parameters
    ----------
    engine : Engine
    threshold : float, default 30.0
        Absolute percentage change required to flag a state.

    Returns
    -------
    tuple[pd.DataFrame, str]
        ``(changes_df, month_analyzed)`` where *changes_df* has columns
        ``customer_state``, ``revenue``, ``prev_revenue``, ``pct_change``.

    Raises
    ------
    ValueError
        If the underlying query returns no data.
    """
    sql = """
        SELECT
            strftime('%Y-%m', order_purchase_timestamp) AS month,
            customer_state,
            ROUND(SUM(price), 2)                        AS revenue
        FROM sales_master
        GROUP BY month, customer_state
        ORDER BY month
    """
    df = run_query(engine, sql)
    if df.empty:
        raise ValueError("No data available to detect significant changes.")

    # Heuristic: drop the latest month if it looks incomplete
    completeness_sql = """
        SELECT
            strftime('%Y-%m', order_purchase_timestamp)     AS month,
            COUNT(DISTINCT DATE(order_purchase_timestamp))  AS days_with_orders
        FROM sales_master
        GROUP BY month
        ORDER BY month DESC
        LIMIT 1
    """
    latest_check = run_query(engine, completeness_sql)
    if not latest_check.empty and latest_check.iloc[0]["days_with_orders"] < 15:
        incomplete_month = latest_check.iloc[0]["month"]
        logger.info("Excluding likely-incomplete month: %s", incomplete_month)
        df = df[df["month"] != incomplete_month]

    df["prev_revenue"] = df.groupby("customer_state")["revenue"].shift(1)
    df["pct_change"] = (
        (df["revenue"] - df["prev_revenue"]) / df["prev_revenue"].replace(0, float("nan"))
    ) * 100

    latest_month = df["month"].max()
    changes = (
        df[
            (df["month"] == latest_month)
            & (df["pct_change"].abs() > threshold)
        ]
        .dropna(subset=["pct_change"])
        .sort_values("pct_change", ascending=False)
    )
    return changes, latest_month


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def get_forecast(
    monthly_data: pd.DataFrame,
    periods: int = 3,
) -> pd.DataFrame:
    """
    Fit a Prophet model on monthly revenue and return a forecast frame.

    Weekly and daily seasonality are disabled because the input granularity
    is monthly.  Yearly seasonality is only enabled when the training set
    spans at least two full years, which is the minimum for Prophet to learn
    a reliable annual pattern.

    Parameters
    ----------
    monthly_data : pd.DataFrame
        Must contain columns ``ds`` (datetime) and ``y`` (float).
    periods : int, default 3
        Number of future months to forecast.

    Returns
    -------
    pd.DataFrame
        Prophet forecast frame with columns ``ds``, ``yhat``,
        ``yhat_lower``, ``yhat_upper``.

    Raises
    ------
    ValueError
        If fewer than 6 data points are available.
    """
    validate_dataframe(monthly_data, ["ds", "y"], "get_forecast")
    if len(monthly_data) < 6:
        raise ValueError(
            f"Only {len(monthly_data)} monthly data points available — need at least 6 "
            "for a stable forecast."
        )

    model = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=(len(monthly_data) >= 24),
        interval_width=0.80,
    )
    model.fit(monthly_data)

    future = model.make_future_dataframe(periods=periods, freq="MS")
    return model.predict(future)


def evaluate_forecast_holdout(
    monthly_data: pd.DataFrame,
    holdout_periods: int = 3,
) -> dict:
    """
    Honest out-of-sample evaluation using a simple train/holdout split.

    Trains a Prophet model on all but the last *holdout_periods* months,
    then compares predictions against the actual held-out values.  This gives
    a realistic accuracy estimate instead of an in-sample (overfit) metric.

    Parameters
    ----------
    monthly_data : pd.DataFrame
        Columns: ``ds`` (datetime), ``y`` (float).
    holdout_periods : int, default 3
        Number of trailing months to withhold from training.

    Returns
    -------
    dict
        Keys: ``mape`` (float), ``comparison`` (DataFrame),
        ``n_train_months`` (int), ``n_test_months`` (int).

    Raises
    ------
    ValueError
        If there is not enough data to form a meaningful split.
    """
    from sklearn.metrics import mean_absolute_percentage_error

    if len(monthly_data) <= holdout_periods + 3:
        raise ValueError(
            f"Need at least {holdout_periods + 4} months of data for a "
            f"{holdout_periods}-month holdout evaluation."
        )

    train = monthly_data.iloc[:-holdout_periods]
    test  = monthly_data.iloc[-holdout_periods:]

    model = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        yearly_seasonality=False,
    )
    model.fit(train)

    future   = model.make_future_dataframe(periods=holdout_periods, freq="MS")
    forecast = model.predict(future)

    comparison = test.merge(forecast[["ds", "yhat"]], on="ds")
    mape = mean_absolute_percentage_error(comparison["y"], comparison["yhat"])

    return {
        "mape":           mape,
        "comparison":     comparison,
        "n_train_months": len(train),
        "n_test_months":  len(test),
    }


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """
    Return a cached Groq API client.

    Raises
    ------
    EnvironmentError
        If ``GROQ_API_KEY`` is not set in the environment / .env file.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file in the project root: GROQ_API_KEY=gsk_..."
        )
    return Groq(api_key=api_key)


def _groq_chat(
    client: Groq,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Call the Groq chat-completions endpoint with exponential-backoff retry.

    Retries up to ``_GROQ_MAX_RETRIES`` times on transient API errors
    (rate-limit, timeout, 5xx).  Raises :class:`InsightGenerationError` on
    permanent failure.

    Parameters
    ----------
    client : Groq
    messages : list[dict]
        OpenAI-style message list.
    temperature : float, default 0.0
    max_tokens : int, default 1024

    Returns
    -------
    str
        The assistant's response text, stripped of leading/trailing whitespace.
    """
    last_exc: Exception | None = None
    for attempt in range(_GROQ_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30,
            )
            return response.choices[0].message.content.strip()
        except GroqError as exc:
            wait = _GROQ_BACKOFF_BASE ** attempt
            logger.warning(
                "Groq API error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, _GROQ_MAX_RETRIES, exc, wait,
            )
            last_exc = exc
            time.sleep(wait)

    raise InsightGenerationError(
        f"Groq API failed after {_GROQ_MAX_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )


def get_schema_info() -> str:
    """
    Return a concise schema description for the ``sales_master`` table.

    This string is injected verbatim into LLM prompts so the model can
    generate valid SQLite queries without needing access to the live DB.

    Returns
    -------
    str
        Multi-line schema description.
    """
    return """
Table: sales_master
Description: Delivered e-commerce orders from the Olist Brazilian marketplace.

Columns
-------
order_id                   TEXT     — Unique order identifier
customer_id                TEXT     — Unique customer identifier
order_status               TEXT     — Always 'delivered' (table is pre-filtered)
order_purchase_timestamp   DATETIME — When the order was placed ('YYYY-MM-DD HH:MM:SS')
price                      REAL     — Unit price of one line item (BRL)
freight_value              REAL     — Shipping cost for that line item (BRL)
product_category_name      TEXT     — English category name, e.g. 'health_beauty'
customer_city              TEXT     — Customer's city
customer_state             TEXT     — Two-letter Brazilian state code, e.g. 'SP', 'RJ'
payment_type               TEXT     — 'credit_card', 'boleto', 'voucher', 'debit_card'
payment_value              REAL     — Total payment for the order (BRL)
""".strip()


def is_safe_query(sql: str) -> bool:
    """
    Return True only if *sql* is a plain SELECT with no dangerous keywords.

    This is a defence-in-depth measure.  In a real production deployment the
    LLM should connect via a read-only database user; this check adds a
    second layer in case that constraint is ever loosened.

    Parameters
    ----------
    sql : str
        Raw SQL string to inspect.

    Returns
    -------
    bool
    """
    if not sql or not sql.strip():
        return False

    cleaned = sql.strip().rstrip(";")
    if ";" in cleaned:          # stacked queries
        return False

    upper = cleaned.upper()
    if not upper.startswith("SELECT"):
        return False

    return not any(kw in upper for kw in FORBIDDEN_SQL_KEYWORDS)


def natural_language_to_sql(question: str, client: Groq) -> str:
    """
    Translate a natural-language question into a valid SQLite SELECT query.

    Returns the sentinel string ``'NOT_RELEVANT'`` when the question is
    unrelated to the sales dataset, so the caller can skip DB execution
    without pattern-matching on error messages.

    Parameters
    ----------
    question : str
        User's question in plain English.
    client : Groq
        Authenticated Groq client.

    Returns
    -------
    str
        A SQLite SELECT statement, or ``'NOT_RELEVANT'``.
    """
    schema = get_schema_info()
    prompt = f"""You are an expert SQLite data analyst embedded in a business analytics tool.

SCHEMA
------
{schema}

TASK
----
Convert the user question below into a single valid SQLite SELECT query.

RULES
-----
1. If the question is NOT related to e-commerce sales data (e.g. general knowledge,
   current events, coding help, anything outside the schema above), respond with
   exactly the word: NOT_RELEVANT
2. Return ONLY the raw SQL — no markdown, no backticks, no explanation.
3. Use only columns and tables that exist in the schema above.
4. Always include every aggregated value (SUM, COUNT, AVG, etc.) in the SELECT
   clause — never reference an alias only in ORDER BY.
5. Add LIMIT 20 unless the question clearly asks for a single aggregated value.
6. Only SELECT is allowed. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
   TRUNCATE, ATTACH, PRAGMA, or REPLACE.
7. Use strftime('%Y-%m', order_purchase_timestamp) for month-level grouping.

USER QUESTION: "{question}"
"""
    raw = _groq_chat(client, [{"role": "user", "content": prompt}], temperature=0)
    # Strip any accidental markdown code fences
    sql = raw.replace("```sql", "").replace("```", "").strip()
    return sql


def ask_data_question(
    question: str,
    engine: Engine,
    client: Groq,
) -> dict:
    """
    End-to-end pipeline: natural-language question → SQL → DB result.

    Steps
    -----
    1. Translate question to SQL via LLM.
    2. Check whether the question is relevant to the dataset.
    3. Run the allowlist safety check.
    4. Execute the query against the live database.
    5. Return a structured result dict consumed by ``app.py``.

    Parameters
    ----------
    question : str
    engine : Engine
    client : Groq

    Returns
    -------
    dict
        Always contains ``status`` (str), ``sql`` (str | None),
        ``data`` (DataFrame | None), ``message`` (str | None).
        ``status`` is one of: ``"ok"``, ``"not_relevant"``, ``"unsafe"``,
        ``"error"``.
    """
    try:
        sql = natural_language_to_sql(question, client)
    except InsightGenerationError as exc:
        return {
            "status": "error", "sql": None, "data": None,
            "message": f"LLM translation failed: {exc}",
        }

    if sql == "NOT_RELEVANT":
        return {
            "status": "not_relevant", "sql": None, "data": None,
            "message": "Question is unrelated to the sales dataset.",
        }

    if not is_safe_query(sql):
        logger.warning("Blocked unsafe generated query: %.300s", sql)
        return {
            "status": "unsafe", "sql": sql, "data": None,
            "message": "Generated query failed the safety check (only SELECT is allowed).",
        }

    try:
        result = run_query(engine, sql)
    except Exception as exc:
        return {
            "status": "error", "sql": sql, "data": None,
            "message": f"Query execution failed: {exc}",
        }

    return {"status": "ok", "sql": sql, "data": result, "message": None}


def generate_ai_insight(
    changes_df: pd.DataFrame,
    month: str,
    client: Groq,
) -> str:
    """
    Generate a stakeholder-ready business-insight narrative from anomaly data.

    Sends the detected month-over-month revenue changes to the LLM and
    returns a structured bullet-point summary suitable for a C-suite report.

    Parameters
    ----------
    changes_df : pd.DataFrame
        Output of :func:`get_significant_changes`.
    month : str
        Month being analysed, e.g. ``"2018-07"``.
    client : Groq

    Returns
    -------
    str
        Markdown-formatted insight text.
    """
    if changes_df.empty:
        return "No significant month-over-month changes were detected for this period."

    changes_text = changes_df[["customer_state", "revenue", "pct_change"]].to_string(index=False)

    confidence_note = (
        "\n**Note:** Only a small number of states show significant movement this month — "
        "treat conclusions as directional rather than definitive.\n"
        if len(changes_df) < 3 else ""
    )

    prompt = f"""You are a senior business data analyst preparing a monthly performance briefing
for the executive team of a Brazilian e-commerce company.

PERIOD ANALYSED: {month}
METRIC: Month-over-month revenue change (%) by customer state

DATA
----
{changes_text}
{confidence_note}

INSTRUCTIONS
------------
Write a concise executive briefing (5–6 bullet points) covering:
  1. The top 2 high-growth states — with magnitude and a plausible business driver.
  2. Any states showing a significant decline — risk level and recommended monitoring.
  3. Cross-state patterns or structural observations (e.g. regional concentration risk).
  4. One specific, prioritised, data-driven recommendation the business should act on
     within the next 30 days.

Tone: professional, analytical, direct. No filler phrases. No hedging. No markdown headers —
bullet points only. Each bullet must reference at least one specific number from the data.
"""
    return _groq_chat(client, [{"role": "user", "content": prompt}], temperature=0.25)


def generate_category_insight(
    top_categories: pd.DataFrame,
    client: Groq,
) -> str:
    """
    Generate an AI commentary on category-level revenue performance.

    Parameters
    ----------
    top_categories : pd.DataFrame
        Output of :func:`get_top_categories` or :func:`get_product_performance`.
    client : Groq

    Returns
    -------
    str
        Markdown-formatted insight text.
    """
    table = top_categories.to_string(index=False)
    prompt = f"""You are a senior e-commerce category manager.

Below is the top product-category performance table for a Brazilian online marketplace.

{table}

Write a 4-bullet executive commentary covering:
1. The dominant category and what it signals about the customer base.
2. Any category with a disproportionately high or low average item price — and what
   pricing strategy that implies.
3. A freight-to-price ratio observation if available, or order-volume efficiency.
4. One growth opportunity: an underperforming category with potential, backed by the data.

Be specific — reference actual numbers. No filler. No markdown headers.
"""
    return _groq_chat(client, [{"role": "user", "content": prompt}], temperature=0.3)