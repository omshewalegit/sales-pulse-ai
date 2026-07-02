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
- LLM integration: a business-semantics-aware NL -> SQL engine and
  AI business-insight generation via Groq (GPT-OSS 120B), with
  exponential-backoff retry

NL -> SQL engine (v2)
----------------------
The naive "translate English to SQL" approach produces SQL that is
syntactically valid but semantically wrong (e.g. counting `customer_id`
instead of `customer_unique_id`, treating `payment_value` as revenue).
This module fixes that with a multi-stage pipeline:

    1. Ambiguity detection   -- refuse to guess; ask a clarifying question.
    2. Intent detection      -- common analytics questions are answered with
                                 a deterministic, pre-written SQL template.
                                 No LLM call, zero hallucination risk.
    3. LLM generation        -- only for questions that don't match a known
                                 intent. The prompt is loaded with a full
                                 business dictionary and hard business rules.
    4. Business-rule fixes   -- a regex safety net that auto-corrects known
                                 mistakes (customer_id misuse, payment_value
                                 used as revenue) before execution.
    5. Self-review           -- a second, independent LLM pass audits the
                                 SQL against the same rules and can rewrite
                                 it again.
    6. Schema validation     -- rejects SQL referencing unknown tables.
    7. Safety check          -- existing allowlist validator (SELECT-only).
    8. Confidence scoring    -- 0-100, based on how much correction was
                                 required to reach a clean query.
    9. Execution              -- against the live database.

Design principles
-----------------
- Every public function is fully type-annotated and has a NumPy-style docstring.
- Database I/O is centralised through `run_query` so failures are caught,
  logged, and surfaced consistently.
- The Groq client and SQLAlchemy engine are each created once and cached;
  Streamlit reruns do not create duplicate connections.
- No business logic lives in app.py -- only presentation.

Author : Sales Pulse AI
"""

from __future__ import annotations

import logging
import os
import re
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

# Prophet tuning -- validated in 01_data_exploration.ipynb:
#   default changepoint_prior_scale (0.05) -> 35.94% out-of-sample MAPE
#   changepoint_prior_scale=0.5            -> 27.10% out-of-sample MAPE
# This constant is the single source of truth so the production forecast
# and the holdout-accuracy panel always evaluate the SAME model.
FORECAST_CHANGEPOINT_PRIOR_SCALE: float = 0.5

# Monthly trend default window -- matches the notebook's validated range
# (Jan 2017 through Aug 2018 inclusive, i.e. exclusive upper bound of Sep 2018).
DEFAULT_TREND_START: str = "2017-01-01"
DEFAULT_TREND_END: str = "2018-09-01"


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

    Notes
    -----
    ``avg_order_value`` is computed at the ORDER level: line items are first
    summed per ``order_id``, then those order totals are averaged. A naive
    ``AVG(price)`` across line items understates true AOV whenever an order
    contains more than one item (this dataset averages ~1.15 items/order),
    because it is an average of item prices, not order totals.

    ``total_customers`` uses ``customer_unique_id`` (the real-person
    identifier), not ``customer_id`` (which is generated per-order in this
    dataset and would make every customer look unique every time).

    Returns
    -------
    dict
        Keys: ``total_revenue``, ``total_orders``, ``avg_order_value``,
        ``total_customers``, ``top_state``, ``top_category``.
    """
    base_sql = """
        SELECT
            ROUND(SUM(price), 2)               AS total_revenue,
            COUNT(DISTINCT order_id)           AS total_orders,
            COUNT(DISTINCT customer_unique_id) AS total_customers
        FROM sales_master
    """
    base = run_query(engine, base_sql)

    # Order-level AOV: sum items per order first, then average across orders.
    aov_sql = """
        SELECT ROUND(AVG(order_total), 2) AS avg_order_value
        FROM (
            SELECT order_id, SUM(price) AS order_total
            FROM sales_master
            GROUP BY order_id
        )
    """
    aov = run_query(engine, aov_sql)

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
        "avg_order_value":  float(aov.iloc[0]["avg_order_value"] or 0),
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
    start_date: str = DEFAULT_TREND_START,
    end_date: str = DEFAULT_TREND_END,
) -> pd.DataFrame:
    """
    Aggregate revenue by calendar month for time-series forecasting.

    The *end_date* is exclusive. Defaults match the range validated in
    01_data_exploration.ipynb (Jan 2017 through Aug 2018 inclusive — i.e.
    an exclusive upper bound of 2018-09-01). The dataset's last calendar
    month with meaningful order volume is August 2018; excluding it (as an
    earlier default of "2018-08-01" did) silently drops a full month of
    training data and shifts the holdout-evaluation window out of sync with
    the notebook's validated results.

    Parameters
    ----------
    engine : Engine
    start_date : str, default ``"2017-01-01"``
        Inclusive lower bound (ISO 8601 date string).
    end_date : str, default ``"2018-09-01"``
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
    Build a state x month revenue matrix suitable for a heatmap visualisation.

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
    Return the distribution of customer review scores (1-5).

    Returns
    -------
    pd.DataFrame
        Columns: ``review_score``, ``count``, ``share_pct``.
        Returns an empty DataFrame if ``review_score`` is not in the schema
        (the current pipeline does not join ``olist_order_reviews_dataset.csv``
        into ``sales_master`` — this function degrades gracefully rather than
        erroring until that join is added).
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
    changepoint_prior_scale: float = FORECAST_CHANGEPOINT_PRIOR_SCALE,
) -> pd.DataFrame:
    """
    Fit a Prophet model on monthly revenue and return a forecast frame.

    Weekly and daily seasonality are disabled because the input granularity
    is monthly.  Yearly seasonality is only enabled when the training set
    spans at least two full years, which is the minimum for Prophet to learn
    a reliable annual pattern.

    ``changepoint_prior_scale`` defaults to 0.5 rather than Prophet's own
    default of 0.05 — this value was empirically validated in
    01_data_exploration.ipynb, where it reduced out-of-sample MAPE from
    35.94% to 27.10% on a 3-month holdout. It is exposed as a parameter
    (rather than hard-coded) so :func:`evaluate_forecast_holdout` can be
    called with the exact same setting the production model uses, keeping
    the reported accuracy panel honest about what's actually being served.

    Parameters
    ----------
    monthly_data : pd.DataFrame
        Must contain columns ``ds`` (datetime) and ``y`` (float).
    periods : int, default 3
        Number of future months to forecast.
    changepoint_prior_scale : float, default 0.5
        Prophet trend-flexibility parameter. Higher values allow the trend
        to bend more sharply at changepoints.

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
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.80,
    )
    model.fit(monthly_data)

    future = model.make_future_dataframe(periods=periods, freq="MS")
    return model.predict(future)


def evaluate_forecast_holdout(
    monthly_data: pd.DataFrame,
    holdout_periods: int = 3,
    changepoint_prior_scale: float = FORECAST_CHANGEPOINT_PRIOR_SCALE,
) -> dict:
    """
    Honest out-of-sample evaluation using a simple train/holdout split.

    Trains a Prophet model on all but the last *holdout_periods* months,
    then compares predictions against the actual held-out values.  This gives
    a realistic accuracy estimate instead of an in-sample (overfit) metric.

    Uses the same ``changepoint_prior_scale`` as the production model
    (:func:`get_forecast`) by default, so the MAPE shown on the dashboard
    reflects the model that is actually generating the forecast — not a
    different, untuned model.

    Parameters
    ----------
    monthly_data : pd.DataFrame
        Columns: ``ds`` (datetime), ``y`` (float).
    holdout_periods : int, default 3
        Number of trailing months to withhold from training.
    changepoint_prior_scale : float, default 0.5
        Must match the value passed to :func:`get_forecast` for the
        evaluation to be representative of production behaviour.

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
        changepoint_prior_scale=changepoint_prior_scale,
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
# LLM plumbing (shared by insights + NL->SQL)
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
    max_tokens : int, default 2048

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
        except Exception as exc:
            # Non-GroqError transient failures (network hiccups, timeouts
            # raised by the underlying HTTP client) get the same retry
            # treatment instead of bubbling straight up uncaught.
            wait = _GROQ_BACKOFF_BASE ** attempt
            logger.warning(
                "Unexpected error calling Groq (attempt %d/%d): %s — retrying in %.1fs",
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
Grain: one row per (order, line item) — an order with 3 items produces 3 rows
       sharing the same order_id.

Columns
-------
order_id                   TEXT     — Order identifier (repeats across line items)
customer_id                TEXT     — Per-ORDER customer identifier (technical, NOT a person)
customer_unique_id          TEXT     — Per-PERSON customer identifier (the real customer)
order_status                TEXT     — Always 'delivered' (table is pre-filtered)
order_purchase_timestamp    DATETIME — When the order was placed ('YYYY-MM-DD HH:MM:SS')
price                       REAL     — Unit price of one line item (BRL) — this IS revenue
freight_value                REAL     — Shipping cost for that line item (BRL)
product_category_name       TEXT     — English category name, e.g. 'health_beauty'
customer_city                TEXT     — Customer's city
customer_state               TEXT     — Two-letter Brazilian state code, e.g. 'SP', 'RJ'
payment_type                 TEXT     — 'credit_card', 'boleto', 'voucher', 'debit_card'
payment_value                 REAL     — Payment TRANSACTION amount (not the same as revenue)
""".strip()


# ---------------------------------------------------------------------------
# NL -> SQL: Business Dictionary & Semantic Layer
# ---------------------------------------------------------------------------

# The only table/columns the LLM (and the validator) are allowed to reference.
KNOWN_TABLES: frozenset[str] = frozenset({"sales_master"})

KNOWN_COLUMNS: frozenset[str] = frozenset({
    "order_id", "customer_id", "customer_unique_id", "order_status",
    "order_purchase_timestamp", "price", "freight_value",
    "product_category_name", "customer_city", "customer_state",
    "payment_type", "payment_value",
})

BUSINESS_DICTIONARY: str = """
BUSINESS DICTIONARY — sales_master
===================================
order_id                 -> One order. Multiple rows share an order_id when an
                             order has several line items. Use
                             COUNT(DISTINCT order_id) to count orders.
customer_id                -> A TECHNICAL, PER-ORDER identifier. In this dataset
                             every order gets its own customer_id, even for the
                             same real person. NEVER use customer_id to count
                             customers, measure repeat purchases, or do ANY
                             customer-level analytics.
customer_unique_id           -> The REAL PERSON identifier. ALWAYS use this column
                             for unique customer counts, repeat-customer
                             analysis, customer cohorts, or any question about
                             "how many customers".
price                        -> The selling price of ONE line item. This is the
                             business definition of REVENUE in this project:
                                 Revenue = SUM(price)
                             NOT payment_value.
freight_value                 -> Shipping cost for that line item. Not part of revenue.
payment_value                  -> The amount recorded in a PAYMENT TRANSACTION for an
                             order. Can differ from order revenue (installments,
                             rounding, split payments). NEVER use payment_value
                             to answer a "revenue" question — only use it when
                             the user explicitly asks about payments or
                             transaction amounts.
customer_state                -> Two-letter Brazilian state code (customer location).
customer_city                  -> Customer's city.
product_category_name          -> Product category (English translated name).
payment_type                    -> 'credit_card', 'boleto', 'voucher', 'debit_card'.
order_purchase_timestamp        -> When the order was placed. Use
                             strftime('%Y-%m', order_purchase_timestamp) to
                             group by month.
order_status                     -> Always 'delivered' in this table.

AVERAGE ORDER VALUE (AOV) must be computed at the ORDER level — sum items per
order first, then average the order totals:
    SELECT AVG(order_total) FROM (
        SELECT order_id, SUM(price) AS order_total
        FROM sales_master GROUP BY order_id
    )
A flat AVG(price) across line items is WRONG — it averages item prices, not
order totals, and understates AOV whenever orders contain multiple items.
""".strip()

BUSINESS_RULES: str = """
HARD BUSINESS RULES (never violate these):
1. Unique / total customers -> COUNT(DISTINCT customer_unique_id). Never customer_id.
2. Order count -> COUNT(DISTINCT order_id).
3. Repeat customers -> GROUP BY customer_unique_id, HAVING COUNT(DISTINCT order_id) > 1.
4. Any customer-level analytics question -> always customer_unique_id, never customer_id.
5. Revenue -> SUM(price). Never payment_value, unless the question explicitly
   asks about payments or transactions.
6. Average Order Value -> the order-level AOV formula above, never a flat
   AVG(price) across line items.
7. SQL dialect: SQLite only.
8. Never invent tables. The only table available is sales_master.
9. Never invent columns. Only use columns listed in the business dictionary.
10. Always include every aggregated value (SUM, COUNT, AVG, etc.) directly in
    the SELECT clause — never reference an alias only in ORDER BY.
11. Add LIMIT 20 unless the question clearly asks for a single aggregated value.
12. Return ONLY the raw SQL statement. No markdown, no backticks, no explanation.
""".strip()


def _order_level_aov_sql() -> str:
    """SQL template implementing the order-level AOV business rule."""
    return (
        "SELECT ROUND(AVG(order_total), 2) AS avg_order_value FROM ("
        "SELECT order_id, SUM(price) AS order_total FROM sales_master GROUP BY order_id"
        ")"
    )


# ---------------------------------------------------------------------------
# NL -> SQL: Intent templates
#
# Each entry matches a common analytics question via regex and returns a
# pre-written, business-rule-correct SQL template. Templated answers never
# touch the LLM, so they carry zero hallucination risk and get the highest
# confidence score.
# ---------------------------------------------------------------------------

INTENT_TEMPLATES: list[dict] = [
    {
        "name": "unique_customers",
        "patterns": [r"\bunique\s+customers?\b", r"\bhow many customers\b", r"\btotal customers?\b",
                     r"\bnumber of customers\b"],
        "exclude": [r"\brepeat\b", r"\breturning\b", r"\bby\s+(state|month|category)\b"],
        "sql": "SELECT COUNT(DISTINCT customer_unique_id) AS unique_customers FROM sales_master",
        "reasoning": "Matched the 'unique customers' intent template (COUNT DISTINCT customer_unique_id, per business rule).",
    },
    {
        "name": "repeat_customers",
        "patterns": [r"\brepeat customers?\b", r"\breturning customers?\b"],
        "exclude": [r"\bby\s+(state|month|category)\b", r"\btrend\b"],
        "sql": (
            "SELECT COUNT(*) AS repeat_customers FROM ("
            "SELECT customer_unique_id FROM sales_master "
            "GROUP BY customer_unique_id HAVING COUNT(DISTINCT order_id) > 1"
            ")"
        ),
        "reasoning": "Matched the 'repeat customers' intent template (grouped by customer_unique_id).",
    },
    {
        "name": "total_orders",
        "patterns": [r"\btotal (number of )?orders?\b", r"\bhow many orders\b"],
        "exclude": [r"\bby\s+(state|month|category)\b"],
        "sql": "SELECT COUNT(DISTINCT order_id) AS total_orders FROM sales_master",
        "reasoning": "Matched the 'total orders' intent template (COUNT DISTINCT order_id).",
    },
    {
        "name": "total_revenue",
        "patterns": [r"\btotal revenue\b", r"\boverall revenue\b", r"^\s*revenue\s*\??\s*$",
                     r"\bwhat is (the )?revenue\b"],
        "exclude": [r"\bby\b", r"\bper\b", r"\btrend\b", r"\bmonth\b", r"\bcategory\b",
                    r"\bstate\b", r"\bpayment\b"],
        "sql": "SELECT ROUND(SUM(price), 2) AS total_revenue FROM sales_master",
        "reasoning": "Matched the 'total revenue' intent template (SUM(price), per business rule — not payment_value).",
    },
    {
        "name": "avg_order_value",
        "patterns": [r"\baverage order value\b", r"\bavg\.? order value\b", r"\baov\b"],
        "exclude": [],
        "sql": _order_level_aov_sql(),
        "reasoning": "Matched the 'average order value' intent template (order-level AOV subquery, per business rule).",
    },
    {
        "name": "top_categories",
        "patterns": [r"\btop\s*\d*\s*(product )?categor(y|ies)\b", r"\bbest.?selling categor"],
        "exclude": [],
        "sql": (
            "SELECT product_category_name, ROUND(SUM(price), 2) AS total_revenue, "
            "COUNT(DISTINCT order_id) AS total_orders FROM sales_master "
            "GROUP BY product_category_name ORDER BY total_revenue DESC LIMIT {limit}"
        ),
        "reasoning": "Matched the 'top categories' intent template (revenue-ranked category breakdown).",
    },
    {
        "name": "top_states",
        "patterns": [r"\btop\s*\d*\s*states?\b", r"\bstate.*(highest|most) revenue\b",
                     r"\bwhich state\b.*revenue\b"],
        "exclude": [],
        "sql": (
            "SELECT customer_state, ROUND(SUM(price), 2) AS total_revenue, "
            "COUNT(DISTINCT order_id) AS total_orders FROM sales_master "
            "GROUP BY customer_state ORDER BY total_revenue DESC LIMIT {limit}"
        ),
        "reasoning": "Matched the 'top states' intent template (revenue-ranked state breakdown).",
    },
    {
        "name": "monthly_revenue",
        "patterns": [r"\bmonthly revenue\b", r"\brevenue.*(trend|by month|per month)\b",
                     r"\bmonth.*revenue\b"],
        "exclude": [],
        "sql": (
            "SELECT strftime('%Y-%m', order_purchase_timestamp) AS month, "
            "ROUND(SUM(price), 2) AS revenue FROM sales_master "
            "GROUP BY month ORDER BY month LIMIT 24"
        ),
        "reasoning": "Matched the 'monthly revenue trend' intent template (month-grouped SUM(price)).",
    },
    {
        "name": "payment_mix",
        "patterns": [r"\bpayment (type|mix|method)s?\b", r"\bhow (do|does) customers? pay\b"],
        "exclude": [],
        "sql": (
            "SELECT payment_type, ROUND(SUM(price), 2) AS total_revenue, "
            "COUNT(DISTINCT order_id) AS total_orders FROM sales_master "
            "GROUP BY payment_type ORDER BY total_revenue DESC"
        ),
        "reasoning": "Matched the 'payment mix' intent template (revenue by payment_type).",
    },
]


def detect_intent(question: str) -> Optional[dict]:
    """
    Match *question* against known analytics intents.

    Parameters
    ----------
    question : str

    Returns
    -------
    dict | None
        The matching entry from :data:`INTENT_TEMPLATES`, or ``None`` if no
        intent matched (the question should fall through to the LLM).
    """
    q = question.lower()
    for tmpl in INTENT_TEMPLATES:
        matched = any(re.search(p, q) for p in tmpl["patterns"])
        excluded = any(re.search(p, q) for p in tmpl.get("exclude", []))
        if matched and not excluded:
            return tmpl
    return None


def _extract_limit(question: str, default: int = 10) -> int:
    """Pull a 'top N' style limit out of the question text, clamped to [1, 50]."""
    match = re.search(r"\btop\s+(\d+)\b", question.lower())
    if match:
        try:
            return max(1, min(int(match.group(1)), 50))
        except ValueError:
            pass
    return default


# ---------------------------------------------------------------------------
# NL -> SQL: Ambiguity detection
# ---------------------------------------------------------------------------

AMBIGUOUS_PATTERNS: list[dict] = [
    {
        "trigger": r"\brevenue\b",
        "unless": [r"\bby\b", r"\bper\b", r"\btotal\b", r"\boverall\b", r"\btrend\b",
                   r"\bmonth\b", r"\bcategory\b", r"\bstate\b", r"\bpayment\b",
                   r"\bwhat is\b"],
        "clarification": (
            "Which cut of revenue do you want — total revenue, revenue by category, "
            "revenue by state, monthly revenue trend, or revenue by payment type?"
        ),
    },
    {
        "trigger": r"\bcustomers?\b",
        "unless": [r"\bunique\b", r"\btotal\b", r"\brepeat\b", r"\breturning\b",
                   r"\bnew\b", r"\bhow many\b", r"\bby\b", r"\bper\b", r"\btop\b",
                   r"\bstate\b", r"\bnumber of\b"],
        "clarification": (
            "Which angle on customers — unique customer count, repeat customers, "
            "new vs. returning, or customers broken down by state?"
        ),
    },
]


def detect_ambiguity(question: str) -> Optional[str]:
    """
    Flag genuinely underspecified questions instead of letting the LLM guess.

    Parameters
    ----------
    question : str

    Returns
    -------
    str | None
        A clarifying question to show the user, or ``None`` if the question
        is specific enough to proceed.
    """
    q = question.lower()
    for rule in AMBIGUOUS_PATTERNS:
        if re.search(rule["trigger"], q) and not any(re.search(p, q) for p in rule["unless"]):
            return rule["clarification"]
    return None


# ---------------------------------------------------------------------------
# NL -> SQL: Business-rule auto-fix layer (regex safety net)
# ---------------------------------------------------------------------------

def apply_business_rule_fixes(sql: str, question: str) -> tuple[str, list[str]]:
    """
    Auto-correct known business-rule violations in LLM-generated SQL.

    This is a deterministic safety net that runs regardless of whether the
    LLM followed the prompt's business rules. It catches the two most common
    failure modes seen in production: counting ``customer_id`` instead of
    ``customer_unique_id``, and treating ``payment_value`` as revenue.

    Parameters
    ----------
    sql : str
        Raw SQL returned by the LLM.
    question : str
        Original natural-language question (used to decide which fixes apply).

    Returns
    -------
    tuple[str, list[str]]
        ``(corrected_sql, list_of_fix_descriptions)``. The list is empty if
        no fix was needed.
    """
    fixes: list[str] = []
    fixed = sql
    q_lower = question.lower()

    # Rule: customer-level analytics must use customer_unique_id.
    if re.search(r"\bcustomers?\b", q_lower) and re.search(r"\bcustomer_id\b", fixed) \
            and "customer_unique_id" not in fixed:
        fixed = re.sub(r"\bcustomer_id\b", "customer_unique_id", fixed)
        fixes.append("Rewrote customer_id -> customer_unique_id (customer analytics must use the real-person ID).")

    # Rule: "revenue" questions must use price, not payment_value.
    if re.search(r"\brevenue\b", q_lower) and re.search(r"\bpayment_value\b", fixed, re.IGNORECASE):
        fixed = re.sub(r"\bpayment_value\b", "price", fixed, flags=re.IGNORECASE)
        fixes.append("Rewrote payment_value -> price (this project defines revenue as SUM(price)).")

    return fixed, fixes


def validate_sql_schema(sql: str) -> list[str]:
    """
    Check that *sql* only references known tables.

    A full column-level parse is out of scope for a regex-based validator,
    so this focuses on the highest-value, cheapest check: table names.
    Any invented/unknown column will still surface immediately as a clear
    SQLite "no such column" error at execution time, so it is not silently
    swallowed even though this validator doesn't catch it up front.

    Parameters
    ----------
    sql : str

    Returns
    -------
    list[str]
        Human-readable validation issues. Empty if the SQL looks clean.
    """
    issues: list[str] = []
    referenced = set(re.findall(r"\bFROM\s+([A-Za-z_]\w*)", sql, re.IGNORECASE)) | \
                 set(re.findall(r"\bJOIN\s+([A-Za-z_]\w*)", sql, re.IGNORECASE))
    unknown = {t for t in referenced if t.lower() not in KNOWN_TABLES}
    if unknown:
        issues.append(f"References unknown table(s): {', '.join(sorted(unknown))}")
    return issues


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


# ---------------------------------------------------------------------------
# NL -> SQL: LLM generation + self-review
# ---------------------------------------------------------------------------

def natural_language_to_sql(question: str, client: Groq) -> str:
    """
    Translate a natural-language question into a valid SQLite SELECT query.

    The prompt is loaded with the schema, the business dictionary, and the
    hard business rules so the model is told the MEANING of each column, not
    just its name — this is what fixes semantically-wrong-but-syntactically-
    valid SQL (e.g. COUNT(customer_id) for "unique customers").

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
    prompt = f"""You are an expert SQLite data analyst embedded in a business analytics tool
for a Brazilian e-commerce company (the Olist dataset).

SCHEMA
------
{schema}

{BUSINESS_DICTIONARY}

{BUSINESS_RULES}

TASK
----
Convert the user question below into a single valid SQLite SELECT query,
strictly following the business dictionary and hard rules above. If the
question is NOT related to e-commerce sales data (general knowledge, current
events, coding help, anything outside this schema), respond with exactly:
NOT_RELEVANT

USER QUESTION: "{question}"
"""
    raw = _groq_chat(client, [{"role": "user", "content": prompt}], temperature=0)
    sql = raw.replace("```sql", "").replace("```", "").strip()
    return sql


def self_review_sql(sql: str, question: str, client: Groq) -> tuple[str, bool]:
    """
    Second, independent LLM pass that audits SQL against the business rules.

    Runs as a business-rule sanity check separate from the generation call —
    a fresh pass is more likely to catch a violation than asking the same
    call to grade its own work. If Groq is unavailable, this degrades
    gracefully to a no-op rather than blocking the whole pipeline.

    Parameters
    ----------
    sql : str
        SQL to review (ideally already passed through
        :func:`apply_business_rule_fixes`).
    question : str
        Original natural-language question, for context.
    client : Groq

    Returns
    -------
    tuple[str, bool]
        ``(possibly_corrected_sql, was_changed)``.
    """
    review_prompt = f"""You are a strict SQL reviewer for a SQLite analytics database.

{BUSINESS_DICTIONARY}

{BUSINESS_RULES}

USER QUESTION: "{question}"

SQL TO REVIEW:
{sql}

Check ONLY for business-rule violations (not style or formatting). If the SQL
is fully compliant, respond with exactly: OK
If it violates a rule, respond with ONLY the corrected SQL — no markdown, no
explanation, no commentary.
"""
    try:
        raw = _groq_chat(client, [{"role": "user", "content": review_prompt}], temperature=0)
    except InsightGenerationError as exc:
        logger.warning("Self-review call failed, proceeding without it: %s", exc)
        return sql, False

    cleaned = raw.replace("```sql", "").replace("```", "").strip()
    if cleaned.upper() == "OK" or not cleaned:
        return sql, False
    if cleaned.rstrip(";").strip().upper() == sql.rstrip(";").strip().upper():
        return sql, False
    return cleaned, True


def compute_confidence(
    source: str,
    business_fixes_applied: list[str],
    review_changed: bool,
    schema_issues: list[str],
) -> int:
    """
    Score how much correction was required to reach the final SQL.

    Deterministic templates start at the highest confidence since they carry
    zero hallucination risk by construction. LLM-generated SQL starts lower
    and is further penalised for each correction stage that had to intervene
    — every auto-fix or self-review rewrite is evidence the first-pass
    output was wrong, so more correction implies lower confidence in
    whatever mistakes might remain uncaught.

    Parameters
    ----------
    source : str
        ``"template"`` or ``"llm"``.
    business_fixes_applied : list[str]
        Fixes applied by :func:`apply_business_rule_fixes`.
    review_changed : bool
        Whether :func:`self_review_sql` rewrote the query.
    schema_issues : list[str]
        Issues from :func:`validate_sql_schema` (non-empty forces 0).

    Returns
    -------
    int
        Confidence score, 0-100.
    """
    if schema_issues:
        return 0

    score = 95 if source == "template" else 70
    if business_fixes_applied:
        score -= 15
    if review_changed:
        score -= 15
    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# NL -> SQL: Orchestrator
# ---------------------------------------------------------------------------

def ask_data_question(
    question: str,
    engine: Engine,
    client: Groq,
) -> dict:
    """
    End-to-end pipeline: natural-language question -> validated SQL -> result.

    See the module docstring for the full 9-stage pipeline description.

    Parameters
    ----------
    question : str
    engine : Engine
    client : Groq

    Returns
    -------
    dict
        Always contains: ``status``, ``sql``, ``data``, ``message``,
        ``intent``, ``confidence``, ``reasoning``, ``source``.
        ``status`` is one of: ``"ok"``, ``"ambiguous"``, ``"not_relevant"``,
        ``"unsafe"``, ``"error"``.
    """
    result: dict = {
        "status": None, "sql": None, "data": None, "message": None,
        "intent": None, "confidence": None, "reasoning": None, "source": None,
    }

    # 1. Ambiguity check — never guess a business-critical breakdown.
    clarification = detect_ambiguity(question)
    if clarification:
        return {**result, "status": "ambiguous", "message": clarification}

    business_fixes: list[str] = []
    review_changed = False

    # 2. Intent detection — deterministic templates for common questions.
    intent = detect_intent(question)
    if intent:
        sql = intent["sql"]
        if "{limit}" in sql:
            sql = sql.format(limit=_extract_limit(question))
        source = "template"
        reasoning = intent["reasoning"]
    else:
        # 3. LLM generation, prompt-loaded with business dictionary + rules.
        try:
            sql = natural_language_to_sql(question, client)
        except InsightGenerationError as exc:
            return {**result, "status": "error", "message": f"LLM translation failed: {exc}"}

        if sql == "NOT_RELEVANT":
            return {**result, "status": "not_relevant",
                    "message": "Question is unrelated to the sales dataset."}

        source = "llm"
        reasoning = "No template matched — SQL generated by the LLM using the injected business dictionary and hard rules."

        # 4. Business-rule auto-fix (deterministic safety net).
        sql, business_fixes = apply_business_rule_fixes(sql, question)
        if business_fixes:
            reasoning += " Auto-corrected: " + " ".join(business_fixes)

        # 5. Self-review (independent second LLM pass).
        try:
            sql, review_changed = self_review_sql(sql, question, client)
            if review_changed:
                reasoning += " Self-review flagged and rewrote a rule violation."
        except Exception as exc:
            logger.warning("Self-review step failed, proceeding with pre-review SQL: %s", exc)

    # 6. Schema validation — only sales_master may be referenced.
    schema_issues = validate_sql_schema(sql)
    if schema_issues:
        return {
            **result, "status": "error", "sql": sql, "source": source,
            "message": "Schema validation failed: " + "; ".join(schema_issues),
        }

    # 7. Safety check — SELECT-only allowlist.
    if not is_safe_query(sql):
        logger.warning("Blocked unsafe generated query: %.300s", sql)
        return {
            **result, "status": "unsafe", "sql": sql, "source": source,
            "message": "Generated query failed the safety check (only SELECT is allowed).",
        }

    # 8. Confidence score.
    confidence = compute_confidence(source, business_fixes, review_changed, schema_issues)

    # 9. Execute.
    try:
        data = run_query(engine, sql)
    except Exception as exc:
        return {
            **result, "status": "error", "sql": sql, "source": source,
            "confidence": confidence, "reasoning": reasoning,
            "intent": intent["name"] if intent else None,
            "message": f"Query execution failed: {exc}",
        }

    return {
        "status": "ok", "sql": sql, "data": data, "message": None,
        "intent": intent["name"] if intent else "llm_generated",
        "confidence": confidence, "reasoning": reasoning, "source": source,
    }


# ---------------------------------------------------------------------------
# LLM: business insight generation (unchanged behaviour, reuses _groq_chat)
# ---------------------------------------------------------------------------

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
Write a concise executive briefing (5-6 bullet points) covering:
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