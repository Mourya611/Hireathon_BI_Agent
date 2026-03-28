import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import parse, request

import duckdb
import pandas as pd


def _extract_sql(text: str) -> str:
    fenced = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    text = text.strip()
    if "SELECT" in text.upper():
        return text
    return ""


def _is_read_only_sql(sql: str) -> bool:
    s = sql.strip().strip(";").lower()
    return s.startswith("select") or s.startswith("with")


@dataclass
class QueryResult:
    question: str
    sql: str
    df: pd.DataFrame
    chart_type: str
    chart_explanation: str = ""
    notes: str = ""


class ConversationalBIAgent:
    def __init__(self, data_dir: str = "data", db_path: str = "instacart.duckdb"):
        self.data_dir = data_dir
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.last_generation_source = "heuristic"
        self.last_llm_error = ""
        self._init_db()

    def _llm_timeout_seconds(self) -> float:
        try:
            return float(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
        except ValueError:
            return 10.0

    def _should_skip_llm(self, question: str) -> bool:
        q = question.lower()
        unsafe_tokens = (
            "drop ",
            "delete ",
            "update ",
            "insert ",
            "truncate ",
            "alter ",
            "create ",
            "grant ",
            "revoke ",
        )
        heavy_export_tokens = (
            "full export",
            "all joined tables",
            "no limit",
            "everything",
            "export all",
            "dump data",
        )
        metadata_tokens = (
            "list of tables",
            "list tables",
            "show tables",
            "what tables",
            "which tables",
            "show columns",
            "columns of",
            "describe table",
            "row count",
            "rows in each table",
        )
        return any(t in q for t in unsafe_tokens + heavy_export_tokens + metadata_tokens)

    def _init_db(self) -> None:
        base = os.path.abspath(self.data_dir).replace("\\", "/")
        self.conn.execute(
            f"""
            CREATE OR REPLACE VIEW orders AS
            SELECT * FROM read_csv_auto('{base}/orders.csv', header=true);

            CREATE OR REPLACE VIEW order_products__prior AS
            SELECT * FROM read_csv_auto('{base}/order_products__prior.csv', header=true);

            CREATE OR REPLACE VIEW order_products__train AS
            SELECT * FROM read_csv_auto('{base}/order_products__train.csv', header=true);

            CREATE OR REPLACE VIEW products AS
            SELECT * FROM read_csv_auto('{base}/products.csv', header=true);

            CREATE OR REPLACE VIEW aisles AS
            SELECT * FROM read_csv_auto('{base}/aisles.csv', header=true);

            CREATE OR REPLACE VIEW departments AS
            SELECT * FROM read_csv_auto('{base}/departments.csv', header=true);

            CREATE OR REPLACE VIEW order_products_all AS
            SELECT * FROM order_products__prior
            UNION ALL
            SELECT * FROM order_products__train;
            """
        )

    def schema_context(self) -> str:
        return """
Tables and keys:
- orders(order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order)
- order_products__prior(order_id, product_id, add_to_cart_order, reordered)
- order_products__train(order_id, product_id, add_to_cart_order, reordered)
- order_products_all(order_id, product_id, add_to_cart_order, reordered)  -- union of prior+train
- products(product_id, product_name, aisle_id, department_id)
- aisles(aisle_id, aisle)
- departments(department_id, department)

Join paths:
- order_products_all.order_id = orders.order_id
- order_products_all.product_id = products.product_id
- products.aisle_id = aisles.aisle_id
- products.department_id = departments.department_id

Rules:
- Use DuckDB SQL.
- Return a single read-only SELECT query.
- Use LIMIT 1000 unless query asks for full exports.
- Handle first-order NULLs in days_since_prior_order with care.
""".strip()

    def _heuristic_sql(self, question: str, last_sql: Optional[str]) -> str:
        q = question.lower()

        if (
            "list of tables" in q
            or "list tables" in q
            or "show tables" in q
            or "what tables" in q
            or "which tables" in q
        ):
            return """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'main'
ORDER BY table_name
""".strip()

        if (
            "table schema" in q
            or "schema of" in q
            or "columns of" in q
            or "describe table" in q
            or "show columns" in q
        ):
            return """
SELECT
  table_name,
  column_name,
  data_type
FROM information_schema.columns
WHERE table_schema = 'main'
ORDER BY table_name, ordinal_position
LIMIT 1000
""".strip()

        if (
            "row count" in q
            or "count rows" in q
            or "rows in each table" in q
            or "table sizes" in q
        ):
            return """
SELECT 'orders' AS table_name, COUNT(*) AS row_count FROM orders
UNION ALL
SELECT 'order_products__prior' AS table_name, COUNT(*) AS row_count FROM order_products__prior
UNION ALL
SELECT 'order_products__train' AS table_name, COUNT(*) AS row_count FROM order_products__train
UNION ALL
SELECT 'order_products_all' AS table_name, COUNT(*) AS row_count FROM order_products_all
UNION ALL
SELECT 'products' AS table_name, COUNT(*) AS row_count FROM products
UNION ALL
SELECT 'aisles' AS table_name, COUNT(*) AS row_count FROM aisles
UNION ALL
SELECT 'departments' AS table_name, COUNT(*) AS row_count FROM departments
ORDER BY row_count DESC
""".strip()

        if "top department" in q and "reorder" in q:
            return """
SELECT
  d.department,
  AVG(op.reordered)::DOUBLE AS reorder_rate,
  COUNT(*) AS product_events
FROM order_products_all op
JOIN products p ON op.product_id = p.product_id
JOIN departments d ON p.department_id = d.department_id
GROUP BY 1
HAVING COUNT(*) > 1000
ORDER BY reorder_rate DESC
LIMIT 20
""".strip()

        if "top aisle" in q and "reorder" in q and "basket position" in q:
            return """
SELECT
  a.aisle,
  AVG(op.reordered)::DOUBLE AS reorder_rate,
  AVG(op.add_to_cart_order)::DOUBLE AS avg_basket_position,
  COUNT(*) AS product_events
FROM order_products_all op
JOIN products p ON op.product_id = p.product_id
JOIN aisles a ON p.aisle_id = a.aisle_id
GROUP BY 1
HAVING COUNT(*) > 500
ORDER BY reorder_rate DESC
LIMIT 25
""".strip()

        if ("order" in q and "day" in q) or "dow" in q:
            return """
SELECT
  order_dow,
  COUNT(*) AS orders
FROM orders
GROUP BY 1
ORDER BY 1
""".strip()

        if ("hour" in q and "order" in q) or "hour of day" in q:
            return """
SELECT
  order_hour_of_day,
  COUNT(*) AS orders
FROM orders
GROUP BY 1
ORDER BY 1
""".strip()

        if "top product" in q or ("most" in q and "product" in q):
            return """
SELECT
  p.product_name,
  COUNT(*) AS times_ordered,
  AVG(op.reordered)::DOUBLE AS reorder_rate
FROM order_products_all op
JOIN products p ON op.product_id = p.product_id
GROUP BY 1
ORDER BY times_ordered DESC
LIMIT 20
""".strip()

        if last_sql and ("now filter" in q or "filter" in q):
            if "organic" in q:
                return f"""
SELECT *
FROM ({last_sql}) t
WHERE LOWER(COALESCE(product_name, '')) LIKE '%organic%'
LIMIT 1000
""".strip()

        return """
SELECT
  d.department,
  COUNT(*) AS product_events,
  AVG(op.reordered)::DOUBLE AS reorder_rate
FROM order_products_all op
JOIN products p ON op.product_id = p.product_id
JOIN departments d ON p.department_id = d.department_id
GROUP BY 1
ORDER BY product_events DESC
LIMIT 20
""".strip()

    def _llm_sql(
        self,
        question: str,
        history: List[Dict[str, str]],
        error: Optional[str] = None,
    ) -> Optional[str]:
        self.last_llm_error = ""
        if self._should_skip_llm(question):
            self.last_llm_error = "Skipped LLM for safety/latency. Using built-in SQL heuristics."
            return None
        history_text = "\n".join(
            [f"Q: {h['question']}\nSQL: {h['sql']}" for h in history[-4:]]
        )
        prompt = f"""
You are a senior BI SQL analyst for Instacart data.
{self.schema_context()}

Conversation history (latest few turns):
{history_text if history_text else "None"}

Current user question:
{question}

{f"Previous SQL failed with error: {error}" if error else ""}

Return only one DuckDB SQL query in a ```sql fenced block.
Only read-only SELECT/WITH queries.
If aggregation is used, include clear aliases.
Prefer robust joins and null-safe logic.
"""

        gemini_sql = self._gemini_sql(question, history, error=error)
        if gemini_sql:
            return gemini_sql
        gemini_error = self.last_llm_error

        openai_sql, openai_error = self._openai_compatible_sql(
            provider_name="OpenAI",
            api_key_env="OPENAI_API_KEY",
            model_env="OPENAI_MODEL",
            default_model="gpt-5-mini",
            base_url_env=None,
            prompt=prompt,
        )
        if openai_sql:
            return openai_sql

        groq_sql, groq_error = self._openai_compatible_sql(
            provider_name="Groq",
            api_key_env="GROQ_API_KEY",
            model_env="GROQ_MODEL",
            default_model="openai/gpt-oss-120b",
            base_url_env="GROQ_BASE_URL",
            prompt=prompt,
        )
        if groq_sql:
            return groq_sql

        errors = [e for e in [gemini_error, openai_error, groq_error] if e]
        if errors:
            self.last_llm_error = "; ".join(errors)
        else:
            self.last_llm_error = "No GEMINI_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY configured."
        return None

    def _openai_compatible_sql(
        self,
        provider_name: str,
        api_key_env: str,
        model_env: str,
        default_model: str,
        base_url_env: Optional[str],
        prompt: str,
    ) -> Tuple[Optional[str], str]:
        api_key = os.getenv(api_key_env)
        if not api_key:
            return None, ""
        try:
            from openai import OpenAI
        except Exception as e:
            return None, f"{provider_name} SDK import failed: {e}"

        base_url = os.getenv(base_url_env, "").strip() if base_url_env else ""
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        client_kwargs["timeout"] = self._llm_timeout_seconds()
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        try:
            resp = client.responses.create(
                model=os.getenv(model_env, default_model),
                input=prompt,
            )
        except Exception as e:
            return None, f"{provider_name} request failed: {e}"
        text = resp.output_text if hasattr(resp, "output_text") else str(resp)
        sql = _extract_sql(text)
        if not sql:
            return None, f"{provider_name} returned no SQL."
        return sql.strip().strip(";"), ""

    def _openai_sql(
        self,
        question: str,
        history: List[Dict[str, str]],
        error: Optional[str] = None,
    ) -> Optional[str]:
        # Retained for backward compatibility if called externally.
        history_text = "\n".join(
            [f"Q: {h['question']}\nSQL: {h['sql']}" for h in history[-4:]]
        )
        prompt = f"""
You are a senior BI SQL analyst for Instacart data.
{self.schema_context()}

Conversation history (latest few turns):
{history_text if history_text else "None"}

Current user question:
{question}

{f"Previous SQL failed with error: {error}" if error else ""}

Return only one DuckDB SQL query in a ```sql fenced block.
Only read-only SELECT/WITH queries.
If aggregation is used, include clear aliases.
Prefer robust joins and null-safe logic.
"""
        sql, _ = self._openai_compatible_sql(
            provider_name="OpenAI",
            api_key_env="OPENAI_API_KEY",
            model_env="OPENAI_MODEL",
            default_model="gpt-5-mini",
            base_url_env=None,
            prompt=prompt,
        )
        return sql

    def _openai_compatible_text(
        self,
        provider_name: str,
        api_key_env: str,
        model_env: str,
        default_model: str,
        base_url_env: Optional[str],
        prompt: str,
    ) -> Tuple[Optional[str], str]:
        api_key = os.getenv(api_key_env)
        if not api_key:
            return None, ""
        try:
            from openai import OpenAI
        except Exception as e:
            return None, f"{provider_name} SDK import failed: {e}"

        base_url = os.getenv(base_url_env, "").strip() if base_url_env else ""
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        client_kwargs["timeout"] = self._llm_timeout_seconds()
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        try:
            resp = client.responses.create(
                model=os.getenv(model_env, default_model),
                input=prompt,
            )
        except Exception as e:
            return None, f"{provider_name} request failed: {e}"

        text = resp.output_text if hasattr(resp, "output_text") else str(resp)
        cleaned = text.strip()
        if not cleaned:
            return None, f"{provider_name} returned empty text."
        return cleaned, ""

    def _gemini_sql(
        self,
        question: str,
        history: List[Dict[str, str]],
        error: Optional[str] = None,
    ) -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        history_text = "\n".join(
            [f"Q: {h['question']}\nSQL: {h['sql']}" for h in history[-4:]]
        )
        prompt = f"""
You are a senior BI SQL analyst for Instacart data.
{self.schema_context()}

Conversation history (latest few turns):
{history_text if history_text else "None"}

Current user question:
{question}

{f"Previous SQL failed with error: {error}" if error else ""}

Return only one DuckDB SQL query in a ```sql fenced block.
Only read-only SELECT/WITH queries.
If aggregation is used, include clear aliases.
Prefer robust joins and null-safe logic.
"""
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{parse.quote(model, safe='')}:generateContent?key={parse.quote(api_key, safe='')}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self._llm_timeout_seconds()) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as e:
            self.last_llm_error = f"Gemini HTTP error: {e.code}"
            return None
        except (urlerror.URLError, TimeoutError, ValueError) as e:
            self.last_llm_error = f"Gemini request failed: {e}"
            return None

        text_parts: List[str] = []
        for cand in body.get("candidates", []):
            content = cand.get("content", {})
            for part in content.get("parts", []):
                txt = part.get("text")
                if txt:
                    text_parts.append(txt)
        text = "\n".join(text_parts).strip()
        if not text:
            self.last_llm_error = "Gemini returned empty content."
            return None
        sql = _extract_sql(text)
        if not sql:
            self.last_llm_error = "Gemini returned no SQL."
            return None
        return sql.strip().strip(";")

    def _gemini_text(self, prompt: str) -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{parse.quote(model, safe='')}:generateContent?key={parse.quote(api_key, safe='')}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self._llm_timeout_seconds()) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        text_parts: List[str] = []
        for cand in body.get("candidates", []):
            content = cand.get("content", {})
            for part in content.get("parts", []):
                txt = part.get("text")
                if txt:
                    text_parts.append(txt)

        text = "\n".join(text_parts).strip()
        return text or None

    def _local_chart_explanation(self, question: str, df: pd.DataFrame, chart_type: str) -> str:
        if df.empty:
            return "No data was returned for this question, so there is no trend to explain yet."

        row_count = len(df)
        col_names = list(df.columns)
        numeric_cols = [c for c in col_names if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric_cols = [c for c in col_names if c not in numeric_cols]

        lines = [
            f"This {chart_type} chart is based on {row_count} rows and {len(col_names)} columns.",
            f"It answers: '{question}'.",
        ]

        if non_numeric_cols:
            lines.append(f"The categories shown are from '{non_numeric_cols[0]}'.")
        if numeric_cols:
            metric_col = numeric_cols[0]
            avg_val = float(df[metric_col].dropna().mean()) if not df[metric_col].dropna().empty else 0.0
            max_idx = df[metric_col].idxmax() if not df[metric_col].dropna().empty else None
            min_idx = df[metric_col].idxmin() if not df[metric_col].dropna().empty else None
            lines.append(f"The main metric is '{metric_col}', with an average of about {avg_val:.2f}.")
            if max_idx is not None and min_idx is not None:
                max_label = (
                    str(df.loc[max_idx, non_numeric_cols[0]])
                    if non_numeric_cols and non_numeric_cols[0] in df.columns
                    else "the highest point"
                )
                min_label = (
                    str(df.loc[min_idx, non_numeric_cols[0]])
                    if non_numeric_cols and non_numeric_cols[0] in df.columns
                    else "the lowest point"
                )
                lines.append(
                    f"Highest is '{max_label}' and lowest is '{min_label}' for '{metric_col}'."
                )

        lines.append("Use this to spot where performance is strongest and where it may need attention.")
        return "\n".join(lines)

    def explain_chart(self, question: str, df: pd.DataFrame, chart_type: str) -> str:
        if df.empty:
            return self._local_chart_explanation(question, df, chart_type)

        sample_df = df.head(12).copy()
        sample_text = sample_df.to_csv(index=False)
        prompt = f"""
You are a business analyst helping non-technical users.
Write a short, plain-English explanation of the chart in 4-6 sentences.
Avoid SQL terms and avoid technical jargon.
Focus on what the chart means, key highs/lows, and one simple action point.

Question: {question}
Chart type: {chart_type}
Columns: {list(df.columns)}
Rows in full result: {len(df)}
Sample rows (CSV):
{sample_text}
""".strip()

        gemini_text = self._gemini_text(prompt)
        if gemini_text:
            return gemini_text

        openai_text, _ = self._openai_compatible_text(
            provider_name="OpenAI",
            api_key_env="OPENAI_API_KEY",
            model_env="OPENAI_MODEL",
            default_model="gpt-5-mini",
            base_url_env=None,
            prompt=prompt,
        )
        if openai_text:
            return openai_text

        groq_text, _ = self._openai_compatible_text(
            provider_name="Groq",
            api_key_env="GROQ_API_KEY",
            model_env="GROQ_MODEL",
            default_model="openai/gpt-oss-120b",
            base_url_env="GROQ_BASE_URL",
            prompt=prompt,
        )
        if groq_text:
            return groq_text

        return self._local_chart_explanation(question, df, chart_type)

    def generate_sql(
        self,
        question: str,
        history: List[Dict[str, str]],
        error: Optional[str] = None,
    ) -> str:
        last_sql = history[-1]["sql"] if history else None
        llm_sql = self._llm_sql(question, history, error=error)
        if llm_sql:
            self.last_generation_source = "llm"
            sql = llm_sql
        else:
            self.last_generation_source = "heuristic"
            sql = self._heuristic_sql(question, last_sql=last_sql)
        sql = sql.strip().strip(";")
        if not _is_read_only_sql(sql):
            # Guardrail: never execute non-read-only SQL. Fall back to safe heuristics.
            self.last_generation_source = "heuristic"
            self.last_llm_error = "Blocked non-read-only SQL generated by provider."
            safe_sql = self._heuristic_sql(question, last_sql=last_sql).strip().strip(";")
            if not _is_read_only_sql(safe_sql):
                raise ValueError("Generated SQL is not read-only.")
            return safe_sql
        return sql

    def execute_with_retry(
        self, question: str, history: List[Dict[str, str]]
    ) -> Tuple[str, pd.DataFrame, str]:
        sql = self.generate_sql(question, history, error=None)
        notes = ""
        if self.last_generation_source == "heuristic":
            notes = "LLM SQL generation unavailable. Using built-in SQL heuristics."
            if self.last_llm_error:
                notes = f"{notes} Reason: {self.last_llm_error}"
        try:
            df = self.conn.execute(sql).df()
            return sql, df, notes
        except Exception as e:
            retry_sql = self.generate_sql(question, history, error=str(e))
            df = self.conn.execute(retry_sql).df()
            retry_notes = f"Recovered after retry. Initial error: {e}"
            if notes:
                retry_notes = f"{notes} {retry_notes}"
            return retry_sql, df, retry_notes

    def choose_chart(self, question: str, df: pd.DataFrame) -> str:
        if df.empty:
            return "table"

        q = question.lower()
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        col_names = [str(c).lower() for c in df.columns]

        time_words = (
            "trend",
            "over time",
            "timeseries",
            "time series",
            "by day",
            "by hour",
            "daily",
            "weekly",
            "monthly",
        )
        time_col_tokens = ("date", "time", "hour", "day", "week", "month", "year", "dow")
        distribution_words = (
            "distribution",
            "histogram",
            "spread",
            "variance",
            "quantile",
            "percentile",
            "median",
        )
        correlation_words = ("correlation", "correlate", "relationship", "vs", "versus", "against")
        composition_words = ("share", "percentage", "composition", "contribution", "mix")

        if any(word in q for word in time_words) or any(
            tok in c for c in col_names for tok in time_col_tokens
        ):
            if len(df.columns) >= 2 and numeric_cols:
                return "line"

        if any(word in q for word in correlation_words) and len(numeric_cols) >= 2:
            return "scatter"

        if any(word in q for word in distribution_words) and numeric_cols:
            return "histogram"

        if len(non_numeric_cols) >= 1 and len(numeric_cols) >= 1:
            if len(df) <= 8 and any(word in q for word in composition_words):
                return "pie"
            if "rate" in q or "top" in q or "by" in q or "compare" in q:
                return "bar"

        if len(numeric_cols) == 1 and len(non_numeric_cols) == 0 and len(df) >= 20:
            return "histogram"

        if len(df.columns) == 2 and len(numeric_cols) == 1:
            return "bar"

        if len(df.columns) == 2 and len(numeric_cols) == 2:
            return "scatter"

        return "table"

    def ask(self, question: str, history: List[Dict[str, str]]) -> QueryResult:
        sql, df, notes = self.execute_with_retry(question, history)
        chart = self.choose_chart(question, df)
        chart_explanation = self.explain_chart(question=question, df=df, chart_type=chart)
        return QueryResult(
            question=question,
            sql=sql,
            df=df,
            chart_type=chart,
            chart_explanation=chart_explanation,
            notes=notes,
        )
