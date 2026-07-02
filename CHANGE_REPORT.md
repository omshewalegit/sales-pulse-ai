# Sales Pulse AI — Upgrade Report

## Files changed
- `src/data_utils.py` — full rewrite of the NL→SQL engine + 4 bug fixes
- `app.py` — UI updates to surface the new NL→SQL transparency data + 4 bug fixes
- `01_data_exploration.ipynb` — **not changed** (it already contains the correct,
  validated experiment; the bug was that its conclusions weren't carried into
  `data_utils.py`)

---

## Part 1 — NL→SQL engine upgrade (the main ask)

### Problem
SQL was syntactically valid but semantically wrong: `COUNT(customer_id)` instead
of `COUNT(DISTINCT customer_unique_id)`, `SUM(payment_value)` used as "revenue",
inconsistent aggregation logic vs. the dashboard KPIs. Root cause: the old prompt
said *"convert English to SQL"* and gave the model only column names — no
business meaning.

### What changed — `data_utils.py`

| # | Addition | Function(s) | Why |
|---|----------|-------------|-----|
| 1 | **Business dictionary** — full semantic description of every column, injected into every LLM call | `BUSINESS_DICTIONARY` | The model needs to be *told* `customer_id` is order-scoped and `customer_unique_id` is the person, not asked to infer it from the name |
| 2 | **Hard business rules** — 12 non-negotiable rules (revenue = SUM(price), customer analytics = customer_unique_id, etc.) | `BUSINESS_RULES` | Turns "please be careful" into explicit, checkable instructions |
| 3 | **Rewritten system prompt** | `natural_language_to_sql()` | Now includes schema + business dictionary + rules + task instructions, not just "convert this" |
| 4 | **Intent detection + SQL templates** — 9 common question types (unique customers, total revenue, AOV, top categories/states, monthly trend, payment mix, repeat customers, order count) are matched via regex and answered with a **pre-written, verified SQL template** — no LLM call at all | `detect_intent()`, `INTENT_TEMPLATES`, `_extract_limit()` | Zero hallucination risk for the questions people ask most; also faster and cheaper (no API call) |
| 5 | **Ambiguity detection** — bare "show revenue" / "customers" questions are intercepted *before* SQL generation and the user is asked to clarify | `detect_ambiguity()`, `AMBIGUOUS_PATTERNS` | Matches your spec exactly: never guess, ask instead |
| 6 | **Business-rule auto-fix layer** — regex safety net that rewrites `customer_id`→`customer_unique_id` and `payment_value`→`price` in LLM output, for questions where those rules apply | `apply_business_rule_fixes()` | Deterministic correction even if the LLM prompt is somehow ignored |
| 7 | **Schema validator** — rejects SQL referencing any table other than `sales_master` | `validate_sql_schema()`, `KNOWN_TABLES`, `KNOWN_COLUMNS` | Catches invented tables before execution; invented columns still surface immediately as a clear SQLite error at execution time |
| 8 | **Self-review pass** — a second, independent LLM call re-checks the SQL against the same business rules and can rewrite it again | `self_review_sql()` | A fresh pass catches mistakes the generation pass made; more reliable than asking one call to grade its own work |
| 9 | **Confidence scoring (0–100)** — templates start at 95, LLM output starts at 70, each correction stage (auto-fix, self-review rewrite) subtracts 15, any schema violation forces 0 | `compute_confidence()` | Lets the UI flag low-confidence answers instead of presenting everything with equal certainty |
| 10 | **Full orchestrator rewrite** — runs all 9 stages in order and returns `{status, sql, data, message, intent, confidence, reasoning, source}` | `ask_data_question()` | Single entry point, same call signature as before, richer return payload |

### Verified against your exact examples
```
Q: "How many unique customers are there?"
→ SELECT COUNT(DISTINCT customer_unique_id) AS unique_customers FROM sales_master
  (template match, 95% confidence)

Q: "What is the total revenue?"
→ SELECT ROUND(SUM(price), 2) AS total_revenue FROM sales_master
  (template match, 95% confidence — never touches payment_value)

Q: "Which state generated the highest revenue?"
→ SELECT customer_state, ROUND(SUM(price), 2) AS total_revenue,
         COUNT(DISTINCT order_id) AS total_orders
  FROM sales_master GROUP BY customer_state ORDER BY total_revenue DESC LIMIT 10
  (template match — same aggregation logic as the dashboard KPIs, so numbers
  will always agree)
```
Auto-fix layer tested directly on adversarial input:
```
apply_business_rule_fixes("SELECT COUNT(DISTINCT customer_id) FROM sales_master",
                           "How many unique customers are there?")
→ "SELECT COUNT(DISTINCT customer_unique_id) FROM sales_master"

apply_business_rule_fixes("SELECT SUM(payment_value) FROM sales_master",
                           "What is the total revenue?")
→ "SELECT SUM(price) FROM sales_master"
```

### What changed — `app.py` (Section 7, Data Explorer)
- Handles the new `"ambiguous"` status with an inline clarifying question
  instead of running bad SQL.
- New transparency strip above every result: **Template Match / LLM Generated**
  badge, **confidence %** badge (green ≥80, amber 50–79, red <50), and the
  matched **intent** name.
- Shows the **reasoning** string (what template matched, or what auto-fixes /
  self-review changed) above the SQL.
- Example queries updated to showcase the new templated intents.

### Not implemented as literally specified (and why)
- **Full column-level SQL validation** (checking every referenced column
  exists) was scoped down to **table-level** validation. A regex-based SQL
  parser that reliably extracts columns from arbitrary SELECT/subquery/CTE
  structures is a false-precision trap — it's easy to write one that's wrong
  in ways that block valid queries. An invented column still fails immediately
  and loudly as a SQLite `no such column` error at execution time (returned in
  `message`), so nothing is silently swallowed — it's just caught one step
  later than a full parser would catch it.

---

## Part 2 — Bug fixes carried over from the earlier review

| # | Bug | Fix | File |
|---|-----|-----|------|
| 1 | `get_forecast()` used Prophet's default `changepoint_prior_scale` (0.05) instead of the tuned value (0.5) validated in the notebook — 35.94% MAPE instead of 27.10% | Added `FORECAST_CHANGEPOINT_PRIOR_SCALE = 0.5` constant, used by both `get_forecast()` and `evaluate_forecast_holdout()` so the reported accuracy always matches the model actually serving the forecast | `data_utils.py` |
| 2 | `get_monthly_trend()` default `end_date="2018-08-01"` excluded August 2018, one month short of the notebook's validated `< '2018-09-01'` range | Changed default to `2018-09-01` | `data_utils.py` |
| 3 | `get_kpi_summary()` computed "Avg Order Value" as `AVG(price)` across line items (item-level), not true order-level AOV | Added order-level subquery: sum items per `order_id`, then average the order totals | `data_utils.py` |
| 4 | Broken UI string: *"...AI briefing powered by ."* | Fixed to *"...powered by Groq GPT-OSS 120B."* | `app.py` |
| 5 | KPI sub-label said "Unique customer IDs" for a metric that (correctly) counts `customer_unique_id`, which was misleading given `customer_id` is order-scoped | Updated to "Unique real people (customer_unique_id)"; AOV sub-label updated to describe the new order-level calculation | `app.py` |
| 6 | Section 6 (anomalies) re-ran `get_significant_changes()` a second time when the "AI Executive Briefing" tab was opened | Computed once above the tabs, both tabs reuse the same result | `app.py` |
| 7 | Redundant `except (DataValidationError, Exception)` in Section 2 | Split into two explicit `except` blocks | `app.py` |
| 8 | `_groq_chat()` only retried on `GroqError`; other transient exceptions (network timeouts, etc.) bypassed the retry/backoff entirely | Added a second `except Exception` branch with the same backoff logic | `data_utils.py` |

### Left as-is (documented, not fixed — flag if you want these addressed too)
- `get_delivery_metrics()` and `get_revenue_heatmap_data()` remain unused by
  `app.py`. Harmless, but dead code — no dashboard section currently calls them.
- `get_review_score_distribution()` will always return an empty DataFrame
  because `01_data_exploration.ipynb` never loads or joins
  `olist_order_reviews_dataset.csv` into `sales_master`. It degrades
  gracefully rather than erroring, but it can't produce real output until
  that join is added to the notebook.
- 381 rows with `freight_value <= 0` and 1,102 rows above the 99th-percentile
  price are flagged in the notebook but never filtered anywhere downstream —
  this looked intentional ("flag only, don't remove yet") so it was left
  untouched.

---

## Expected accuracy impact
- **NL→SQL correctness**: the three failure modes in your bug report
  (customer_id misuse, payment_value-as-revenue, inconsistent state-revenue
  aggregation) are now structurally prevented for the 9 templated intents
  (deterministic, can't drift) and actively corrected for everything else via
  the auto-fix + self-review layers.
- **Forecast accuracy**: production MAPE goes from **35.94% → 27.10%** on the
  same 3-month holdout, because the tuned Prophet setting is now actually used
  instead of silently falling back to the default.
- **AOV metric**: no longer systematically understated for any order with
  more than one line item (~15% of orders in this dataset, based on the
  110,197 rows / 96,478 delivered orders ratio in the pipeline output).