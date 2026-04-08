import os
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from bi_agent import ConversationalBIAgent, DatasetProfile


def _load_local_env() -> None:
    if load_dotenv is None:
        return

    project_dir = Path(__file__).resolve().parent
    env_file = project_dir / ".env"
    env_example_file = project_dir / ".env.example"

    loaded = load_dotenv(env_file, override=False) if env_file.exists() else False
    if not loaded and env_example_file.exists():
        load_dotenv(env_example_file, override=False)


def _ensure_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_dataset_label" not in st.session_state:
        st.session_state.active_dataset_label = "Bundled Instacart dataset"


def _runtime_dir() -> Path:
    runtime = Path(__file__).resolve().parent / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return runtime


def _session_dir() -> Path:
    session_root = _runtime_dir() / st.session_state.session_id
    session_root.mkdir(parents=True, exist_ok=True)
    return session_root


def _save_uploaded_files(uploaded_files) -> list[str]:
    upload_dir = _session_dir() / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for uploaded_file in uploaded_files:
        destination = upload_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(str(destination))
    return saved_paths


_load_local_env()

st.set_page_config(page_title="Live Conversational BI", layout="wide")
_ensure_session_state()


@st.cache_resource(show_spinner=False)
def get_agent(session_id: str) -> ConversationalBIAgent:
    session_root = _runtime_dir() / session_id
    session_root.mkdir(parents=True, exist_ok=True)
    db_path = session_root / "workspace.duckdb"
    return ConversationalBIAgent(data_dir="data", db_path=str(db_path))


agent = get_agent(st.session_state.session_id)

st.title("Live Conversational BI Agent")
st.caption(
    "Upload CSV, Parquet, or DuckDB files. Large CSVs are chunked into partitions inside DuckDB "
    "so the app can stay responsive while keeping the same chat-to-SQL workflow."
)
st.markdown(
    "Ask business questions in plain English and get SQL, charts, auto insights, and decision-oriented recommendations."
)

with st.sidebar:
    st.header("Data Setup")
    data_mode = st.radio(
        "Dataset source",
        options=["Use bundled Instacart data", "Upload my files"],
    )
    partition_size_rows = st.number_input(
        "Partition size for CSV rows",
        min_value=50_000,
        max_value=2_000_000,
        value=500_000,
        step=50_000,
        help="Large CSV files are ingested in row-based partitions of this size.",
    )
    uploaded_files = st.file_uploader(
        "Upload dataset files",
        type=["csv", "parquet", "duckdb", "db"],
        accept_multiple_files=True,
        disabled=data_mode != "Upload my files",
    )
    prepare_clicked = st.button("Prepare Dataset", type="primary")
    clear_clicked = st.button("Clear Conversation")

    st.header("LLM Settings")
    gemini_model = st.text_input(
        "Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), key="gemini_model"
    )
    openai_model = st.text_input(
        "OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"), key="openai_model"
    )
    groq_model = st.text_input(
        "Groq model", value=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), key="groq_model"
    )

    if os.getenv("GEMINI_API_KEY"):
        st.success("Gemini key detected.")
    elif os.getenv("OPENAI_API_KEY"):
        st.warning("Gemini key not found. Using OpenAI key fallback.")
    elif os.getenv("GROQ_API_KEY"):
        st.warning("Gemini/OpenAI keys not found. Using Groq key fallback.")
    else:
        st.warning("No API key found. The app will rely on built-in SQL heuristics.")

os.environ["GEMINI_MODEL"] = gemini_model
os.environ["OPENAI_MODEL"] = openai_model
os.environ["GROQ_MODEL"] = groq_model

if clear_clicked:
    st.session_state.history = []
    st.rerun()

if prepare_clicked:
    try:
        with st.spinner("Preparing dataset..."):
            if data_mode == "Use bundled Instacart data":
                profile = agent.load_default_instacart()
            else:
                if not uploaded_files:
                    raise ValueError("Upload at least one CSV, Parquet, or DuckDB file first.")
                saved_paths = _save_uploaded_files(uploaded_files)
                profile = agent.register_data_files(
                    saved_paths,
                    dataset_label="User uploaded dataset",
                    partition_size_rows=int(partition_size_rows),
                )
        if not profile.is_ready:
            raise ValueError("No supported tables were loaded.")
        st.session_state.history = []
        st.session_state.active_dataset_label = profile.dataset_label
        st.success(f"Prepared {len(profile.tables)} table(s) for analysis.")
    except Exception as exc:
        st.error(f"Dataset preparation failed: {exc}")

profile: DatasetProfile = agent.dataset_profile

if profile.is_ready:
    st.subheader("Active Dataset")
    st.write(profile.dataset_label)
    summary_rows = [
        {
            "table": table.table_name,
            "source": table.source_name,
            "format": table.source_format,
            "rows": table.row_count,
            "partitioned": "Yes" if table.partitioned else "No",
            "partitions": table.partition_count,
        }
        for table in profile.tables
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    if profile.relationships:
        st.markdown("### Detected Relationships")
        relationship_rows = [
            {
                "left_table": rel.left_table,
                "left_column": rel.left_column,
                "right_table": rel.right_table,
                "right_column": rel.right_column,
                "confidence": rel.confidence,
                "reason": rel.reason,
            }
            for rel in profile.relationships
        ]
        st.dataframe(pd.DataFrame(relationship_rows), use_container_width=True)
    for note in profile.notes:
        st.info(note)
else:
    st.info("Prepare a dataset from the sidebar to start querying.")

question = st.text_input(
    "Your question",
    placeholder="Example: Show top categories by sales, or list the tables in my upload",
    disabled=not profile.is_ready,
)

col_run, col_reset_dataset = st.columns([1, 1])
run_clicked = col_run.button("Run Query", type="primary", disabled=not profile.is_ready)
reset_dataset_clicked = col_reset_dataset.button("Reset Dataset")

if reset_dataset_clicked:
    agent.reset_dataset()
    st.session_state.history = []
    st.rerun()


def _pick_xy(df: pd.DataFrame):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None, None
    y = numeric_cols[0]
    x_candidates = [c for c in df.columns if c != y]
    x = x_candidates[0] if x_candidates else df.columns[0]
    return x, y


def _pick_two_numeric(df: pd.DataFrame):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return None, None
    return numeric_cols[0], numeric_cols[1]


if run_clicked and question.strip():
    try:
        with st.spinner("Generating SQL and running query..."):
            result = agent.ask(question.strip(), st.session_state.history)
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        st.stop()

    st.session_state.history.append(
        {
            "question": result.question,
            "sql": result.sql,
            "rows": len(result.df),
            "columns": list(result.df.columns),
        }
    )

    if result.notes:
        st.warning(result.notes)

    st.subheader("Generated SQL")
    st.code(result.sql, language="sql")

    st.subheader("Result")
    st.dataframe(result.df, use_container_width=True)

    x, y = _pick_xy(result.df)
    if result.chart_type == "bar" and x and y:
        st.subheader("Chart (Bar)")
        st.bar_chart(result.df.set_index(x)[y])
    elif result.chart_type == "line" and x and y:
        st.subheader("Chart (Line)")
        st.line_chart(result.df.set_index(x)[y])
    elif result.chart_type == "pie" and x and y:
        st.subheader("Chart (Pie)")
        fig, ax = plt.subplots(figsize=(6, 6))
        pie_df = result.df[[x, y]].dropna().head(8)
        ax.pie(pie_df[y], labels=pie_df[x], autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)
    elif result.chart_type == "histogram":
        st.subheader("Chart (Histogram)")
        numeric_cols = [c for c in result.df.columns if pd.api.types.is_numeric_dtype(result.df[c])]
        if numeric_cols:
            hist_col = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(result.df[hist_col].dropna(), bins=20)
            ax.set_xlabel(hist_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("Histogram suggested, but no numeric column was available.")
    elif result.chart_type == "scatter":
        st.subheader("Chart (Scatter)")
        x_num, y_num = _pick_two_numeric(result.df)
        if x_num and y_num:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(result.df[x_num], result.df[y_num], alpha=0.7)
            ax.set_xlabel(x_num)
            ax.set_ylabel(y_num)
            st.pyplot(fig)
        else:
            st.info("Scatter suggested, but at least two numeric columns are required.")
    else:
        st.info("Showing table only for this query shape.")

    if result.chart_reason:
        st.markdown("### Why This Chart")
        st.write(result.chart_reason)

    if result.chart_explanation:
        st.markdown("### What This Means (Plain English)")
        st.write(result.chart_explanation)

    if result.auto_insights:
        st.markdown("### Auto Insights")
        for insight in result.auto_insights:
            st.markdown(f"- {insight}")

    if result.business_suggestions:
        st.markdown("### Business Suggestions")
        for suggestion in result.business_suggestions:
            st.markdown(f"- {suggestion}")

if st.session_state.history:
    st.subheader("Conversation Memory")
    for i, h in enumerate(reversed(st.session_state.history[-8:]), start=1):
        st.markdown(f"**{i}. Q:** {h['question']}")
        st.code(h["sql"], language="sql")
