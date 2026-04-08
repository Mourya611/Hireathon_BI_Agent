import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
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
    if "SELECT" in text.upper() or "WITH" in text.upper():
        return text
    return ""


def _is_read_only_sql(sql: str) -> bool:
    s = sql.strip().strip(";").lower()
    return s.startswith("select") or s.startswith("with")


def _sanitize_identifier(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "dataset"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class QueryResult:
    question: str
    sql: str
    df: pd.DataFrame
    chart_type: str
    chart_explanation: str = ""
    notes: str = ""


@dataclass
class TableProfile:
    table_name: str
    source_name: str
    source_format: str
    row_count: int
    columns: List[str]
    partitioned: bool
    partition_count: int
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    temporal_columns: List[str] = field(default_factory=list)
    sample_values: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RelationshipProfile:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    confidence: str
    reason: str


@dataclass
class DatasetProfile:
    dataset_label: str
    tables: List[TableProfile] = field(default_factory=list)
    relationships: List[RelationshipProfile] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return bool(self.tables)


class ConversationalBIAgent:
    def __init__(self, data_dir: str = "data", db_path: str = "instacart.duckdb"):
        self.data_dir = data_dir
        self.db_path = db_path
        db_parent = Path(db_path).resolve().parent
        db_parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._attached_aliases: List[str] = []
        self.last_generation_source = "heuristic"
        self.last_llm_error = ""
        self.dataset_profile = DatasetProfile(dataset_label="No dataset loaded")

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

    def reset_dataset(self) -> None:
        for alias in self._attached_aliases:
            self.conn.execute(f"DETACH IF EXISTS {alias}")
        self._attached_aliases = []
        objects = self.conn.execute(
            """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()
        for table_name, table_type in objects:
            quoted = f'"{table_name}"'
            if str(table_type).upper() == "VIEW":
                self.conn.execute(f"DROP VIEW IF EXISTS {quoted}")
            else:
                self.conn.execute(f"DROP TABLE IF EXISTS {quoted}")
        self.dataset_profile = DatasetProfile(dataset_label="No dataset loaded")

    def load_default_instacart(self) -> DatasetProfile:
        data_dir = Path(self.data_dir)
        expected_files = [
            "orders.csv",
            "order_products__prior.csv",
            "order_products__train.csv",
            "products.csv",
            "aisles.csv",
            "departments.csv",
        ]
        if not all((data_dir / name).exists() for name in expected_files):
            return self.dataset_profile
        return self.register_data_files(
            [str(data_dir / name) for name in expected_files],
            dataset_label="Bundled Instacart dataset",
            partition_size_rows=750_000,
        )

    def register_data_files(
        self,
        file_paths: List[str],
        dataset_label: str = "Uploaded dataset",
        partition_size_rows: int = 500_000,
    ) -> DatasetProfile:
        self.reset_dataset()
        tables: List[TableProfile] = []
        notes: List[str] = []

        for file_path in file_paths:
            path = Path(file_path)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                tables.append(self._ingest_csv_partitioned(path, partition_size_rows))
            elif suffix == ".parquet":
                tables.append(self._ingest_parquet(path))
            elif suffix in {".duckdb", ".db"}:
                attached_tables = self._ingest_duckdb(path)
                tables.extend(attached_tables)
            else:
                notes.append(f"Skipped unsupported file: {path.name}")

        if not tables:
            self.dataset_profile = DatasetProfile(
                dataset_label=dataset_label,
                tables=[],
                relationships=[],
                notes=notes or ["No supported tables were loaded."],
            )
            return self.dataset_profile

        relationships = self._infer_relationships(tables)
        self.dataset_profile = DatasetProfile(
            dataset_label=dataset_label,
            tables=tables,
            relationships=relationships,
            notes=notes,
        )
        return self.dataset_profile

    def _ingest_csv_partitioned(self, path: Path, partition_size_rows: int) -> TableProfile:
        base_name = _sanitize_identifier(path.stem)
        part_names: List[str] = []
        total_rows = 0
        columns: List[str] = []

        for part_index, chunk in enumerate(pd.read_csv(path, chunksize=partition_size_rows), start=1):
            part_name = f"{base_name}__part_{part_index:04d}"
            self.conn.register("chunk_df", chunk)
            self.conn.execute(f'CREATE OR REPLACE TABLE "{part_name}" AS SELECT * FROM chunk_df')
            self.conn.unregister("chunk_df")
            part_names.append(part_name)
            total_rows += len(chunk)
            if not columns:
                columns = [str(col) for col in chunk.columns]

        if not part_names:
            raise ValueError(f"No rows were read from {path.name}")

        if len(part_names) == 1:
            self.conn.execute(f'CREATE OR REPLACE VIEW "{base_name}" AS SELECT * FROM "{part_names[0]}"')
        else:
            union_sql = "\nUNION ALL\n".join(
                [f'SELECT * FROM "{part_name}"' for part_name in part_names]
            )
            self.conn.execute(f'CREATE OR REPLACE VIEW "{base_name}" AS {union_sql}')

        return self._build_table_profile(
            table_name=base_name,
            source_name=path.name,
            source_format="csv",
            row_count=total_rows,
            partitioned=len(part_names) > 1,
            partition_count=len(part_names),
        )

    def _ingest_parquet(self, path: Path) -> TableProfile:
        base_name = _sanitize_identifier(path.stem)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{base_name}" AS SELECT * FROM read_parquet(?)',
            [str(path)],
        )
        row_count = int(self.conn.execute(f'SELECT COUNT(*) FROM "{base_name}"').fetchone()[0])
        return self._build_table_profile(
            table_name=base_name,
            source_name=path.name,
            source_format="parquet",
            row_count=row_count,
            partitioned=False,
            partition_count=1,
        )

    def _ingest_duckdb(self, path: Path) -> List[TableProfile]:
        alias = _sanitize_identifier(f"src_{path.stem}")
        self.conn.execute(f"ATTACH ? AS {alias} (READ_ONLY)", [str(path)])
        self._attached_aliases.append(alias)
        rows = self.conn.execute(
            f"""
            SELECT table_name
            FROM {alias}.information_schema.tables
            WHERE table_schema = 'main' AND table_type IN ('BASE TABLE', 'VIEW')
            ORDER BY table_name
            """
        ).fetchall()

        profiles: List[TableProfile] = []
        for (table_name,) in rows:
            local_name = _sanitize_identifier(table_name)
            self.conn.execute(
                f'CREATE OR REPLACE VIEW "{local_name}" AS SELECT * FROM {alias}."{table_name}"'
            )
            row_count = int(self.conn.execute(f'SELECT COUNT(*) FROM "{local_name}"').fetchone()[0])
            profiles.append(
                self._build_table_profile(
                    table_name=local_name,
                    source_name=f"{path.name}:{table_name}",
                    source_format=path.suffix.lower().lstrip("."),
                    row_count=row_count,
                    partitioned=False,
                    partition_count=1,
                )
            )
        return profiles

    def _build_table_profile(
        self,
        table_name: str,
        source_name: str,
        source_format: str,
        row_count: int,
        partitioned: bool,
        partition_count: int,
    ) -> TableProfile:
        pragma_rows = self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = [str(row[1]) for row in pragma_rows]
        typed_rows = [(str(row[1]), str(row[2]).upper()) for row in pragma_rows]

        numeric_markers = ("INT", "DECIMAL", "DOUBLE", "FLOAT", "REAL", "HUGEINT", "BIGINT", "SMALLINT")
        temporal_markers = ("DATE", "TIME", "TIMESTAMP")

        numeric_columns = [name for name, dtype in typed_rows if any(marker in dtype for marker in numeric_markers)]
        temporal_columns = [name for name, dtype in typed_rows if any(marker in dtype for marker in temporal_markers)]
        categorical_columns = [
            name for name, dtype in typed_rows if name not in numeric_columns and name not in temporal_columns
        ][:8]

        sample_values: Dict[str, List[str]] = {}
        sample_targets = (categorical_columns[:3] + temporal_columns[:1])[:4]
        for column_name in sample_targets:
            safe_column = f'"{column_name}"'
            rows = self.conn.execute(
                f"""
                SELECT DISTINCT CAST({safe_column} AS VARCHAR)
                FROM "{table_name}"
                WHERE {safe_column} IS NOT NULL
                LIMIT 3
                """
            ).fetchall()
            sample_values[column_name] = [str(row[0]) for row in rows if row and row[0] is not None]

        return TableProfile(
            table_name=table_name,
            source_name=source_name,
            source_format=source_format,
            row_count=row_count,
            columns=columns,
            partitioned=partitioned,
            partition_count=partition_count,
            numeric_columns=numeric_columns[:10],
            categorical_columns=categorical_columns,
            temporal_columns=temporal_columns[:6],
            sample_values=sample_values,
        )

    def _infer_relationships(self, tables: List[TableProfile]) -> List[RelationshipProfile]:
        relationships: List[RelationshipProfile] = []
        seen_pairs = set()

        for left in tables:
            left_cols = set(left.columns)
            for right in tables:
                if left.table_name >= right.table_name:
                    continue
                right_cols = set(right.columns)
                shared = sorted(left_cols.intersection(right_cols))

                for column_name in shared:
                    if not (column_name.endswith("_id") or column_name == "id"):
                        continue
                    confidence = "high" if column_name.endswith("_id") else "medium"
                    key = (left.table_name, column_name, right.table_name, column_name)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    relationships.append(
                        RelationshipProfile(
                            left_table=left.table_name,
                            left_column=column_name,
                            right_table=right.table_name,
                            right_column=column_name,
                            confidence=confidence,
                            reason="Shared key-like column name",
                        )
                    )

                for left_column in left.columns:
                    if not left_column.endswith("_id"):
                        continue
                    stem = left_column[:-3]
                    candidate_columns = {f"{stem}_id", "id"}
                    singular_right = right.table_name.rstrip("s")
                    plural_stem = stem.rstrip("s")
                    if stem not in (right.table_name, singular_right, plural_stem):
                        continue
                    for right_column in right.columns:
                        if right_column in candidate_columns:
                            key = (left.table_name, left_column, right.table_name, right_column)
                            if key in seen_pairs:
                                continue
                            seen_pairs.add(key)
                            relationships.append(
                                RelationshipProfile(
                                    left_table=left.table_name,
                                    left_column=left_column,
                                    right_table=right.table_name,
                                    right_column=right_column,
                                    confidence="medium",
                                    reason="Foreign-key style name matches table name",
                                )
                            )
                            break

        return relationships[:20]

    def schema_context(self) -> str:
        if not self.dataset_profile.tables:
            return "No dataset is loaded yet."

        lines = [f"Dataset: {self.dataset_profile.dataset_label}", "Tables and columns:"]
        for table in self.dataset_profile.tables[:20]:
            column_preview = ", ".join(table.columns[:20])
            lines.append(f"- {table.table_name}({column_preview})")
            lines.append(f"  rows={table.row_count}, numeric={table.numeric_columns[:4]}, temporal={table.temporal_columns[:3]}")
            if table.sample_values:
                sample_bits = [
                    f"{col}: {', '.join(values)}" for col, values in table.sample_values.items() if values
                ]
                if sample_bits:
                    lines.append(f"  samples={'; '.join(sample_bits[:3])}")

        if self.dataset_profile.relationships:
            lines.append("Likely joins:")
            for rel in self.dataset_profile.relationships[:12]:
                lines.append(
                    f"- {rel.left_table}.{rel.left_column} = {rel.right_table}.{rel.right_column} "
                    f"({rel.confidence}; {rel.reason})"
                )

        lines.append("Rules:")
        lines.append("- Use DuckDB SQL.")
        lines.append("- Return a single read-only SELECT query.")
        lines.append("- Use LIMIT 1000 unless the user clearly asks for more.")
        lines.append("- Prefer explicit joins when keys are obvious from *_id columns.")
        return "\n".join(lines)

    def _has_instacart_schema(self) -> bool:
        table_names = {table.table_name for table in self.dataset_profile.tables}
        required = {
            "orders",
            "order_products_prior",
            "order_products_train",
            "products",
            "aisles",
            "departments",
        }
        return required.issubset(table_names)

    def _find_table_profile(self, table_name: str) -> Optional[TableProfile]:
        for table in self.dataset_profile.tables:
            if table.table_name == table_name:
                return table
        return None

    def _question_tokens(self, question: str) -> List[str]:
        return [token for token in _normalize_token(question).split() if token]

    def _score_table_for_question(self, table: TableProfile, question_tokens: List[str]) -> int:
        haystacks = [table.table_name] + table.columns
        score = 0
        for token in question_tokens:
            for item in haystacks:
                normalized_item = _normalize_token(item)
                if token == normalized_item:
                    score += 6
                elif token in normalized_item.split():
                    score += 4
                elif token in normalized_item:
                    score += 2
        if table.row_count > 0:
            score += min(5, len(str(table.row_count)))
        return score

    def _best_table_for_question(self, question: str) -> TableProfile:
        tokens = self._question_tokens(question)
        scored_tables = [
            (self._score_table_for_question(table, tokens), table.row_count, table)
            for table in self.dataset_profile.tables
        ]
        scored_tables.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored_tables[0][2]

    def _best_matching_columns(
        self,
        question: str,
        columns: List[str],
        prefer_numeric: bool = False,
        prefer_temporal: bool = False,
    ) -> List[str]:
        tokens = self._question_tokens(question)
        scored: List[Tuple[int, str]] = []
        for column in columns:
            normalized = _normalize_token(column)
            score = 0
            for token in tokens:
                if token == normalized:
                    score += 8
                elif token in normalized.split():
                    score += 5
                elif token in normalized:
                    score += 2
            if prefer_numeric and any(word in normalized for word in ("amount", "sales", "revenue", "price", "qty", "count", "total")):
                score += 3
            if prefer_temporal and any(word in normalized for word in ("date", "time", "day", "month", "year")):
                score += 3
            scored.append((score, column))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [column for score, column in scored if score > 0]

    def _generic_heuristic_sql(self, question: str) -> str:
        table = self._best_table_for_question(question)
        q = question.lower()

        if any(word in q for word in ("first rows", "sample", "preview", "show rows", "head")):
            return f'SELECT * FROM "{table.table_name}" LIMIT 20'

        matched_temporal = self._best_matching_columns(question, table.temporal_columns, prefer_temporal=True)
        matched_numeric = self._best_matching_columns(question, table.numeric_columns, prefer_numeric=True)
        matched_categorical = self._best_matching_columns(question, table.categorical_columns)

        temporal_col = matched_temporal[0] if matched_temporal else (table.temporal_columns[0] if table.temporal_columns else None)
        numeric_col = matched_numeric[0] if matched_numeric else (table.numeric_columns[0] if table.numeric_columns else None)
        category_col = (
            matched_categorical[0]
            if matched_categorical
            else (table.categorical_columns[0] if table.categorical_columns else None)
        )

        if temporal_col and any(word in q for word in ("trend", "over time", "daily", "monthly", "weekly", "by date")):
            metric_sql = f'SUM("{numeric_col}")' if numeric_col else "COUNT(*)"
            metric_alias = f"total_{numeric_col}" if numeric_col else "row_count"
            return f'''
SELECT
  "{temporal_col}" AS period,
  {metric_sql} AS {metric_alias}
FROM "{table.table_name}"
GROUP BY 1
ORDER BY 1
LIMIT 1000
'''.strip()

        if category_col and any(word in q for word in ("top", "by", "group", "breakdown", "compare")):
            metric_sql = f'SUM("{numeric_col}")' if numeric_col and any(word in q for word in ("sum", "sales", "revenue", "total")) else "COUNT(*)"
            metric_alias = f"total_{numeric_col}" if metric_sql.startswith("SUM") else "row_count"
            return f'''
SELECT
  "{category_col}" AS category,
  {metric_sql} AS {metric_alias}
FROM "{table.table_name}"
WHERE "{category_col}" IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC
LIMIT 20
'''.strip()

        if numeric_col and any(word in q for word in ("average", "avg", "mean", "sum", "total", "max", "min")):
            aggregate = "AVG"
            if any(word in q for word in ("sum", "total")):
                aggregate = "SUM"
            elif "max" in q:
                aggregate = "MAX"
            elif "min" in q:
                aggregate = "MIN"
            return f'''
SELECT
  {aggregate}("{numeric_col}") AS {aggregate.lower()}_{numeric_col}
FROM "{table.table_name}"
'''.strip()

        relationship = self.dataset_profile.relationships[0] if self.dataset_profile.relationships else None
        if relationship and any(word in q for word in ("join", "combine", "across")):
            left_profile = self._find_table_profile(relationship.left_table)
            right_profile = self._find_table_profile(relationship.right_table)
            display_col = None
            if right_profile and right_profile.categorical_columns:
                display_col = right_profile.categorical_columns[0]
            if display_col:
                return f'''
SELECT
  r."{display_col}" AS category,
  COUNT(*) AS row_count
FROM "{relationship.left_table}" l
JOIN "{relationship.right_table}" r
  ON l."{relationship.left_column}" = r."{relationship.right_column}"
GROUP BY 1
ORDER BY 2 DESC
LIMIT 20
'''.strip()

        return f'SELECT * FROM "{table.table_name}" LIMIT 20'

    def _heuristic_sql(self, question: str, last_sql: Optional[str]) -> str:
        q = question.lower()
        table_names = [table.table_name for table in self.dataset_profile.tables]

        if not table_names:
            raise ValueError("No dataset is loaded. Upload data before asking questions.")

        if any(token in q for token in ("list of tables", "list tables", "show tables", "what tables")):
            return """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'main'
ORDER BY table_name
""".strip()

        if any(
            token in q
            for token in ("table schema", "schema of", "columns of", "describe table", "show columns")
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

        if any(token in q for token in ("row count", "count rows", "rows in each table", "table sizes")):
            union_sql = "\nUNION ALL\n".join(
                [
                    f"SELECT '{table_name}' AS table_name, COUNT(*) AS row_count FROM \"{table_name}\""
                    for table_name in table_names
                ]
            )
            return f"{union_sql}\nORDER BY row_count DESC"

        if self._has_instacart_schema():
            if "top department" in q and "reorder" in q:
                return """
SELECT
  d.department,
  AVG(op.reordered)::DOUBLE AS reorder_rate,
  COUNT(*) AS product_events
FROM order_products_prior op
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
FROM order_products_prior op
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
FROM order_products_prior op
JOIN products p ON op.product_id = p.product_id
GROUP BY 1
ORDER BY times_ordered DESC
LIMIT 20
""".strip()

            if last_sql and ("now filter" in q or "filter" in q) and "organic" in q:
                return f"""
SELECT *
FROM ({last_sql}) t
WHERE LOWER(COALESCE(product_name, '')) LIKE '%organic%'
LIMIT 1000
""".strip()

        return self._generic_heuristic_sql(question)

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

        history_text = "\n".join([f"Q: {h['question']}\nSQL: {h['sql']}" for h in history[-4:]])
        prompt = f"""
You are a senior BI SQL analyst.
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
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": self._llm_timeout_seconds()}
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
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": self._llm_timeout_seconds()}
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
        history_text = "\n".join([f"Q: {h['question']}\nSQL: {h['sql']}" for h in history[-4:]])
        prompt = f"""
You are a senior BI SQL analyst.
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
            values = df[metric_col].dropna()
            avg_val = float(values.mean()) if not values.empty else 0.0
            max_idx = values.idxmax() if not values.empty else None
            min_idx = values.idxmin() if not values.empty else None
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
        if not self.dataset_profile.is_ready:
            raise ValueError("No dataset is prepared yet.")
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
