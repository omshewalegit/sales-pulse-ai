"""
app.py
======
Streamlit dashboard for Sales Pulse AI.

Sections
--------
0  KPI strip          — headline numbers pulled from get_kpi_summary()
1  Revenue Breakdown  — top categories + top states (side-by-side tables)
2  Product Deep-Dive  — multi-metric category table + AI category commentary
3  Payment Mix        — payment-method bar chart + share table
4  Customer Cohorts   — new vs returning customers over time
5  Revenue Forecast   — Prophet 3-month forecast + holdout accuracy panel
6  State Anomalies    — month-over-month change detection + AI insights
7  Data Explorer      — natural-language → SQL → live results, with a
                        transparency panel (intent, confidence, reasoning)

Design: all business logic lives in data_utils.py; this file is pure presentation.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_utils import (
    DataValidationError,
    InsightGenerationError,
    ask_data_question,
    evaluate_forecast_holdout,
    generate_ai_insight,
    generate_category_insight,
    get_customer_cohort_data,
    get_engine,
    get_forecast,
    get_groq_client,
    get_kpi_summary,
    get_monthly_trend,
    get_payment_breakdown,
    get_product_performance,
    get_significant_changes,
    get_state_revenue,
    get_top_categories,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sales Pulse AI",
    layout="wide",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB GLOBAL THEME
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "#161B27",
    "axes.facecolor":    "#161B27",
    "axes.edgecolor":    "#1E2738",
    "axes.labelcolor":   "#64748B",
    "axes.titlecolor":   "#E2E8F0",
    "axes.titlesize":    11,
    "axes.titleweight":  "600",
    "axes.labelsize":    9,
    "axes.grid":         True,
    "grid.color":        "#1E2738",
    "grid.linewidth":    0.8,
    "grid.alpha":        1.0,
    "xtick.color":       "#64748B",
    "ytick.color":       "#64748B",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.frameon":    False,
    "legend.labelcolor": "#94A3B8",
    "text.color":        "#E2E8F0",
    "font.family":       "sans-serif",
    "font.size":         9,
    "lines.linewidth":   2,
    "patch.linewidth":   0,
})

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base:      #0F1117;
    --bg-card:      #161B27;
    --bg-card-alt:  #1C2236;
    --accent:       #6C8EF5;
    --accent-dim:   rgba(108,142,245,0.12);
    --accent-glow:  rgba(108,142,245,0.25);
    --positive:     #10B981;
    --warning:      #F59E0B;
    --danger:       #EF4444;
    --text-primary: #E2E8F0;
    --text-muted:   #64748B;
    --text-faint:   #334155;
    --border:       rgba(255,255,255,0.06);
    --border-hover: rgba(108,142,245,0.35);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
section[data-testid="stSidebar"] {
    background-color: var(--bg-base) !important;
}
[data-testid="stMainBlockContainer"] {
    padding: 0 2rem 4rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Nav bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.nav-logo { display: flex; align-items: baseline; gap: 0.4rem; }
.nav-logo-primary { font-size: 1.1rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; }
.nav-logo-accent  { font-size: 1.1rem; font-weight: 700; color: var(--accent);       letter-spacing: -0.02em; }
.nav-badge {
    font-size: 0.62rem; font-weight: 700; color: var(--accent);
    background: var(--accent-dim); border: 1px solid rgba(108,142,245,0.3);
    padding: 0.15rem 0.45rem; border-radius: 3px;
    letter-spacing: 0.09em; text-transform: uppercase; margin-left: 0.4rem;
}
.nav-meta { font-size: 0.73rem; color: var(--text-muted); letter-spacing: 0.02em; display:flex; align-items:center; gap:0.75rem; }
.status-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.18rem 0.6rem; border-radius: 20px;
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.status-live { background: rgba(16,185,129,0.1); color: #10B981; border: 1px solid rgba(16,185,129,0.25); }
.status-dot  { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

/* ── KPI strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.kpi-cell { background: var(--bg-card); padding: 1.2rem 1.4rem; }
.kpi-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.11em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 1.55rem; font-weight: 700; color: var(--text-primary);
    letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.2rem;
}
.kpi-sub { font-size: 0.7rem; color: var(--text-faint); }
.kpi-value-sm {
    font-size: 1rem; font-weight: 700; color: var(--text-primary);
    letter-spacing: -0.01em; line-height: 1.2; margin-bottom: 0.2rem;
}

/* ── Section headers ── */
.section-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.05rem; font-weight: 600; color: var(--text-primary);
    margin-bottom: 0.1rem; letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.78rem; color: var(--text-muted);
    margin-bottom: 1.2rem; line-height: 1.55;
}
.col-label {
    font-size: 0.67rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.55rem;
}

/* ── Divider ── */
.section-divider { border: none; border-top: 1px solid var(--border); margin: 2.5rem 0; }

/* ── Insight block ── */
.insight-block {
    background: var(--bg-card); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 8px;
    padding: 1.4rem 1.6rem; font-size: 0.855rem; line-height: 1.8;
    color: var(--text-primary);
}

/* ── Result banner ── */
.result-banner {
    display: flex; align-items: center; gap: 0.75rem;
    background: rgba(16,185,129,0.07); border: 1px solid rgba(16,185,129,0.18);
    border-radius: 6px; padding: 0.65rem 1rem; margin-bottom: 0.75rem;
}
.result-count { font-size: 1rem; font-weight: 700; color: var(--positive); }
.result-label { font-size: 0.78rem; color: var(--text-muted); }

/* ── NL→SQL transparency panel ── */
.nlsql-meta-strip {
    display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
    margin-bottom: 0.85rem;
}
.nlsql-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.22rem 0.65rem; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em;
    border: 1px solid var(--border);
}
.badge-template { background: rgba(16,185,129,0.1); color: #10B981; border-color: rgba(16,185,129,0.25); }
.badge-llm      { background: rgba(108,142,245,0.1); color: var(--accent); border-color: rgba(108,142,245,0.3); }
.badge-conf-high { background: rgba(16,185,129,0.1);  color: #10B981;  border-color: rgba(16,185,129,0.25); }
.badge-conf-mid  { background: rgba(245,158,11,0.1);  color: #F59E0B;  border-color: rgba(245,158,11,0.25); }
.badge-conf-low  { background: rgba(239,68,68,0.1);   color: #EF4444;  border-color: rgba(239,68,68,0.25); }
.nlsql-reasoning {
    font-size: 0.78rem; color: var(--text-muted); line-height: 1.6;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.7rem 0.9rem; margin-bottom: 0.85rem;
}

/* ── Streamlit overrides ── */
[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }
[data-testid="stDataFrame"] th {
    background: var(--bg-card-alt) !important; color: var(--text-muted) !important;
    font-size: 0.68rem !important; font-weight: 700 !important;
    letter-spacing: 0.09em !important; text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
    background: var(--bg-card) !important; color: var(--text-primary) !important;
    font-size: 0.8rem !important; border-bottom: 1px solid var(--border) !important;
}
[data-testid="stButton"] > button {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important; padding: 0.55rem 1.4rem !important;
    transition: opacity 0.15s !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.82 !important; }
[data-testid="stTextInput"] input {
    background: var(--bg-card) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.875rem !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stCode"] {
    background: #0A0D14 !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}
[data-testid="stExpander"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary { font-size: 0.79rem !important; color: var(--text-muted) !important; font-weight: 500 !important; }
[data-testid="stAlert"] { border-radius: 6px !important; font-size: 0.82rem !important; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    padding: 0.5rem 1.1rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    letter-spacing: 0.03em !important;
    background: transparent !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: var(--text-primary) !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:focus,
[data-testid="stTabs"] [role="tab"]:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}
/* Override Streamlit's BaseWeb tab highlight bar (the red/blue line) */
[data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
    height: 2px !important;
}
[data-baseweb="tab-border"] {
    background-color: var(--border) !important;
    height: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ENGINE INIT
# ══════════════════════════════════════════════════════════════════════════════
try:
    engine = get_engine()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_brl(value: float) -> str:
    """Format a BRL value as a human-readable string."""
    if value >= 1_000_000:
        return f"R$ {value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"R$ {value/1_000:.1f}K"
    return f"R$ {value:.2f}"


def _section(label: str, title: str, sub: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def _col_label(text: str) -> None:
    st.markdown(f'<p class="col-label">{text}</p>', unsafe_allow_html=True)


def _divider() -> None:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">
        <span class="nav-logo-primary">Sales</span>
        <span class="nav-logo-accent">Pulse</span>
        <span class="nav-badge">AI</span>
    </div>
    <div class="nav-meta">
        Olist Brazilian E-Commerce &nbsp;&middot;&nbsp; 2016 – 2018
        <span class="status-pill status-live">
            <span class="status-dot"></span> Live
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
try:
    kpi = get_kpi_summary(engine)
    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-cell">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">{_fmt_brl(kpi["total_revenue"])}</div>
            <div class="kpi-sub">All delivered orders (sum of item price)</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{kpi["total_orders"]:,}</div>
            <div class="kpi-sub">Unique order IDs</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Avg Order Value</div>
            <div class="kpi-value">{_fmt_brl(kpi["avg_order_value"])}</div>
            <div class="kpi-sub">Per order (item totals summed, then averaged)</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{kpi["total_customers"]:,}</div>
            <div class="kpi-sub">Unique real people (customer_unique_id)</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Top State</div>
            <div class="kpi-value-sm">{kpi["top_state"]}</div>
            <div class="kpi-sub">Highest revenue state</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">Top Category</div>
            <div class="kpi-value-sm">{kpi["top_category"].replace("_", " ").title()}</div>
            <div class="kpi-sub">Highest revenue category</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
except Exception as exc:
    logger.exception(exc)

_divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — REVENUE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
_section(
    "Performance Overview", "Revenue Breakdown",
    "Top-performing product categories and geographic markets ranked by total revenue.",
)

col1, col2 = st.columns(2, gap="large")
with col1:
    _col_label("Top 10 Product Categories")
    try:
        st.dataframe(get_top_categories(engine), width="stretch", height=340)
    except Exception as exc:
        st.error("Failed to load category data.")
        logger.exception(exc)

with col2:
    _col_label("Top 10 States by Revenue")
    try:
        st.dataframe(get_state_revenue(engine), width="stretch", height=340)
    except Exception as exc:
        st.error("Failed to load state data.")
        logger.exception(exc)

_divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PRODUCT DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
_section(
    "Product Intelligence", "Category Deep-Dive",
    "Multi-metric performance table: revenue, order volume, average item price, and "
    "freight-to-price ratio. Use the AI commentary to surface strategic observations.",
)

try:
    prod_df = get_product_performance(engine, limit=15)
    st.dataframe(prod_df, width="stretch", height=380)

    try:
        client = get_groq_client()
        if st.button("Generate Category Commentary"):
            with st.spinner("Analysing category performance..."):
                commentary = generate_category_insight(prod_df, client)
            st.markdown(f'<div class="insight-block">{commentary}</div>', unsafe_allow_html=True)
    except EnvironmentError as exc:
        st.error(str(exc))

except DataValidationError as exc:
    st.error("Failed to load product performance data.")
    logger.exception(exc)
except Exception as exc:
    st.error("Failed to load product performance data.")
    logger.exception(exc)

_divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PAYMENT MIX
# ══════════════════════════════════════════════════════════════════════════════
_section(
    "Transaction Intelligence", "Payment Method Breakdown",
    "Revenue and order distribution by payment type. "
    "Reveals customer preference and potential checkout optimisation levers.",
)

try:
    pay_df = get_payment_breakdown(engine)

    # Drop rows where payment_type is null/NaN — can happen if the join
    # produces unmatched rows in some DB builds
    pay_df = pay_df.dropna(subset=["payment_type"])
    pay_df["payment_type"] = pay_df["payment_type"].astype(str)
    pay_df["total_revenue"] = pd.to_numeric(pay_df["total_revenue"], errors="coerce").fillna(0)
    pay_df["revenue_share_pct"] = pd.to_numeric(pay_df["revenue_share_pct"], errors="coerce").fillna(0)
    pay_df = pay_df[pay_df["total_revenue"] > 0].reset_index(drop=True)

    col_chart, col_table = st.columns([3, 2], gap="large")

    with col_chart:
        _col_label("Revenue by Payment Type")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = ["#6C8EF5", "#10B981", "#F59E0B", "#EF4444"]
        labels = pay_df["payment_type"].str.replace("_", " ").str.title().tolist()
        values = pay_df["total_revenue"].tolist()
        shares = pay_df["revenue_share_pct"].tolist()

        bars = ax.barh(
            labels,
            values,
            color=colors[: len(pay_df)],
            height=0.55,
        )
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"R${x/1e6:.1f}M" if x >= 1e6 else f"R${x/1e3:.0f}K")
        )
        for bar, val in zip(bars, shares):
            ax.text(
                bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=8, color="#94A3B8",
            )
        ax.set_xlabel("Total Revenue (BRL)", labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()
        fig.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col_table:
        _col_label("Summary Table")
        st.dataframe(pay_df, width="stretch", height=240)

except Exception as exc:
    st.error("Failed to load payment data.")
    logger.exception(exc)

_divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CUSTOMER COHORTS
# ══════════════════════════════════════════════════════════════════════════════
_section(
    "Customer Analytics", "New vs Returning Customers",
    "Monthly cohort split: customers placing their first-ever order (new) "
    "versus customers who had ordered in a prior month (returning).",
)

try:
    cohort_df = get_customer_cohort_data(engine)
    if not cohort_df.empty:
        cohort_df["month"] = cohort_df["month"].astype(str)

        fig, ax = plt.subplots(figsize=(12, 3.8))
        x = np.arange(len(cohort_df))
        width = 0.42

        ax.bar(x - width / 2, cohort_df["new_customers"],      width, label="New",       color="#6C8EF5", alpha=0.9)
        ax.bar(x + width / 2, cohort_df["returning_customers"], width, label="Returning", color="#10B981", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(cohort_df["month"], rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Customers", labelpad=8)
        ax.legend(loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    else:
        st.info("No cohort data available.")

except Exception as exc:
    st.error("Failed to load cohort data.")
    logger.exception(exc)

_divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
_section(
    "Predictive Analytics", "3-Month Revenue Forecast",
    "Prophet time-series model trained on monthly revenue aggregates "
    "(changepoint_prior_scale=0.5, tuned in notebook — 27.1% out-of-sample MAPE "
    "vs. 35.9% with Prophet's default settings). "
    "Dashed line = forecast; shaded band = 80% confidence interval. "
    "Dotted vertical = train/forecast boundary.",
)

try:
    monthly_data = get_monthly_trend(engine)
    forecast     = get_forecast(monthly_data)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(monthly_data["ds"], monthly_data["y"],
            color="#6C8EF5", linewidth=2.2, zorder=3, label="Actual revenue")
    ax.scatter(monthly_data["ds"], monthly_data["y"],
               color="#6C8EF5", s=34, zorder=4)

    ax.plot(forecast["ds"], forecast["yhat"],
            color="#10B981", linewidth=2, linestyle="--", zorder=3, label="Forecast")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    alpha=0.14, color="#10B981", zorder=2)

    ax.axvline(monthly_data["ds"].iloc[-1], color="#334155", linewidth=1, linestyle=":")

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"R$ {x/1e6:.1f}M" if x >= 1e6 else f"R$ {x/1e3:.0f}K")
    )
    ax.set_xlabel("Month", labelpad=8)
    ax.set_ylabel("Revenue (BRL)", labelpad=8)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    with st.expander("Model Accuracy — Out-of-Sample Holdout Evaluation"):
        st.markdown(
            '<p style="font-size:0.78rem;color:#64748B;line-height:1.6;">'
            "In-sample fit always overstates real performance. The table below "
            "evaluates the model on months it never saw during training, using the "
            "same tuned settings as the production forecast above."
            "</p>", unsafe_allow_html=True,
        )
        try:
            ev = evaluate_forecast_holdout(monthly_data, holdout_periods=3)
            c1, c2, c3 = st.columns(3)
            c1.metric("Out-of-Sample MAPE", f"{ev['mape']*100:.1f}%")
            c2.metric("Training Months",    ev["n_train_months"])
            c3.metric("Test Months",        ev["n_test_months"])
            st.markdown(
                '<p style="font-size:0.74rem;color:#64748B;margin-top:0.5rem;">'
                "Elevated MAPE is expected with ~20 months of history. "
                "Prophet requires 2+ years to learn reliable seasonality."
                "</p>", unsafe_allow_html=True,
            )
            st.dataframe(ev["comparison"], width="stretch")
        except ValueError as exc:
            st.info(str(exc))

except ValueError as exc:
    st.warning(f"Forecast unavailable: {exc}")
except Exception as exc:
    st.error("An error occurred while building the forecast.")
    logger.exception(exc)

_divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — STATE ANOMALIES + AI INSIGHT
# ══════════════════════════════════════════════════════════════════════════════
_section(
    "AI Analysis", "State Revenue Anomalies",
    "Detects states with month-over-month revenue swings exceeding 30%. "
    "Click below to generate an executive-level AI briefing powered by Groq GPT-OSS 120B.",
)

# Computed once and reused by both tabs below, instead of re-querying the
# anomaly-detection SQL a second time when the "AI Executive Briefing" tab
# is opened.
_anomaly_error: Exception | None = None
try:
    _changes, _month_analyzed = get_significant_changes(engine)
except Exception as exc:
    _changes, _month_analyzed = pd.DataFrame(), None
    _anomaly_error = exc
    logger.exception(exc)

tabs = st.tabs(["Anomaly Data", "AI Executive Briefing"])

with tabs[0]:
    if _anomaly_error is not None:
        st.error("Failed to load anomaly data.")
    elif not _changes.empty:
        _col_label(f"Significant Changes — {_month_analyzed}")

        # .applymap() removed in Pandas 2.1+ — use .map() instead
        def _style_pct(val: float):
            try:
                color = "#10B981" if float(val) > 0 else "#EF4444"
                return f"color: {color}; font-weight: 600"
            except (TypeError, ValueError):
                return ""

        display_cols = ["customer_state", "revenue", "prev_revenue", "pct_change"]
        styled = (
            _changes[display_cols]
            .style.map(_style_pct, subset=["pct_change"])
            .format({
                "revenue":      "R$ {:,.2f}",
                "prev_revenue": "R$ {:,.2f}",
                "pct_change":   "{:+.1f}%",
            })
        )
        st.dataframe(styled, width="stretch")
    else:
        st.info("No statistically significant changes detected for the latest period.")

with tabs[1]:
    try:
        client = get_groq_client()
    except EnvironmentError as exc:
        st.error(str(exc))
    else:
        if st.button("Generate AI Briefing"):
            if _anomaly_error is not None:
                st.error("Anomaly data unavailable — cannot generate a briefing.")
            elif _changes.empty:
                st.info("No significant changes to brief on for this period.")
            else:
                with st.spinner("Generating executive briefing..."):
                    try:
                        insight = generate_ai_insight(_changes, _month_analyzed, client)
                        st.markdown(f'<div class="insight-block">{insight}</div>', unsafe_allow_html=True)
                    except InsightGenerationError as exc:
                        st.error(f"Insight generation failed: {exc}")
                    except Exception as exc:
                        st.error("An unexpected error occurred.")
                        logger.exception(exc)

_divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — NATURAL LANGUAGE DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
_section(
    "Data Explorer", "Natural Language SQL",
    "Ask a question in plain English. Common questions are answered with a "
    "verified SQL template; anything else is translated by the LLM using a "
    "business-rule dictionary, then auto-corrected and self-reviewed before "
    "running. Only SELECT queries are permitted.",
)

EXAMPLE_QUERIES = [
    "How many unique customers are there?",
    "What is the total revenue?",
    "Which product category generated the most revenue in SP?",
    "Repeat customers",
    "Top 5 cities by total number of orders",
    "Average order value",
]

st.markdown(
    '<p class="col-label">Example queries</p>', unsafe_allow_html=True
)
st.markdown(
    " &nbsp;&middot;&nbsp; ".join(
        f'<code style="font-size:0.74rem;color:#94A3B8;background:#161B27;'
        f'padding:0.15rem 0.4rem;border-radius:4px;">{q}</code>'
        for q in EXAMPLE_QUERIES
    ),
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

user_question = st.text_input(
    "Question",
    placeholder="e.g. How many unique customers are there?",
    label_visibility="collapsed",
)


def _confidence_badge_class(confidence: int | None) -> str:
    if confidence is None:
        return "badge-conf-mid"
    if confidence >= 80:
        return "badge-conf-high"
    if confidence >= 50:
        return "badge-conf-mid"
    return "badge-conf-low"


if st.button("Run Query"):
    if not user_question.strip():
        st.warning("Enter a question before running the query.")
    else:
        try:
            client = get_groq_client()
        except EnvironmentError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Detecting intent, generating SQL, and validating..."):
                result = ask_data_question(user_question, engine, client)

            if result["status"] == "ambiguous":
                st.info(f"🤔 {result['message']}")

            elif result["status"] == "not_relevant":
                st.warning(
                    "This question does not appear to relate to the sales dataset. "
                    "Try asking about orders, products, revenue, customers, or geographic performance."
                )

            elif result["status"] == "unsafe":
                st.code(result["sql"], language="sql")
                st.error("Query blocked — only SELECT statements are permitted.")

            elif result["status"] == "error":
                if result["sql"]:
                    st.code(result["sql"], language="sql")
                st.error(result["message"])

            else:  # status == "ok"
                source_badge = (
                    '<span class="nlsql-badge badge-template">📋 Template Match</span>'
                    if result["source"] == "template"
                    else '<span class="nlsql-badge badge-llm">🧠 LLM Generated</span>'
                )
                conf = result.get("confidence")
                conf_class = _confidence_badge_class(conf)
                conf_label = f"{conf}% confidence" if conf is not None else "confidence n/a"

                st.markdown(
                    f'<div class="nlsql-meta-strip">{source_badge}'
                    f'<span class="nlsql-badge {conf_class}">✓ {conf_label}</span>'
                    f'<span class="nlsql-badge" style="color:#94A3B8;">'
                    f'intent: {result.get("intent") or "n/a"}</span></div>',
                    unsafe_allow_html=True,
                )

                if result.get("reasoning"):
                    st.markdown(
                        f'<div class="nlsql-reasoning">{result["reasoning"]}</div>',
                        unsafe_allow_html=True,
                    )

                st.code(result["sql"], language="sql")
                st.markdown(
                    f'<div class="result-banner">'
                    f'<span class="result-count">{len(result["data"])}</span>'
                    f'<span class="result-label">rows returned</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(result["data"], width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:3.5rem;padding-top:1.5rem;border-top:1px solid rgba(255,255,255,0.05);
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
    <span style="font-size:0.7rem;color:#334155;">
        Sales Pulse AI &nbsp;&mdash;&nbsp; Analytics Platform
    </span>
    <span style="font-size:0.7rem;color:#334155;">
        Python &middot; SQLite &middot; Prophet &middot; Groq GPT-OSS 120B &middot;
        Olist Brazilian E-Commerce Dataset (Kaggle)
    </span>
</div>
""", unsafe_allow_html=True)