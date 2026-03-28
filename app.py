import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from bi_agent import ConversationalBIAgent


def _load_local_env() -> None:
    if load_dotenv is None:
        return

    project_dir = Path(__file__).resolve().parent
    env_file = project_dir / ".env"
    env_example_file = project_dir / ".env.example"

    loaded = load_dotenv(env_file, override=False) if env_file.exists() else False
    if not loaded and env_example_file.exists():
        load_dotenv(env_example_file, override=False)


_load_local_env()

st.set_page_config(page_title="Instacart Conversational BI", layout="wide")
st.title("Instacart Conversational BI Agent")
st.caption(
    "Ask plain-English questions. The agent generates SQL over all 6 CSV tables, "
    "executes it in DuckDB, and returns charts + tables."
)

with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("Data directory", value="data")
    gemini_model = st.text_input(
        "Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), key="gemini_model"
    )
    openai_model = st.text_input(
        "OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"), key="openai_model"
    )
    groq_model = st.text_input(
        "Groq model", value=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), key="groq_model"
    )
    st.info(
        "Provider order: GEMINI_API_KEY -> OPENAI_API_KEY -> GROQ_API_KEY -> built-in SQL heuristics."
    )
    if os.getenv("GEMINI_API_KEY"):
        st.success("Gemini key detected.")
    elif os.getenv("OPENAI_API_KEY"):
        st.warning("Gemini key not found. Using OpenAI key fallback.")
    elif os.getenv("GROQ_API_KEY"):
        st.warning("Gemini/OpenAI keys not found. Using Groq key fallback.")
    else:
        st.warning("No API key found. Using built-in SQL heuristics.")
    st.markdown("### Example prompts")
    st.markdown("- Top departments by reorder rate")
    st.markdown("- Orders by hour of day")
    st.markdown("- Show aisles with highest reorder rate and avg basket position")
    st.markdown("- Top products by times ordered and reorder rate")

# Keep selected model names active for this Streamlit session.
os.environ["GEMINI_MODEL"] = gemini_model
os.environ["OPENAI_MODEL"] = openai_model
os.environ["GROQ_MODEL"] = groq_model


@st.cache_resource(show_spinner=False)
def get_agent(_data_dir: str) -> ConversationalBIAgent:
    return ConversationalBIAgent(data_dir=_data_dir, db_path="instacart.duckdb")


agent = get_agent(data_dir)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input(
    "Your question",
    placeholder="Example: Which departments have the highest reorder rate?",
)

col_run, col_clear = st.columns([1, 1])
run_clicked = col_run.button("Run Query", type="primary")
clear_clicked = col_clear.button("Clear Conversation")

if clear_clicked:
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
    except Exception as e:
        st.error(f"Query failed: {e}")
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
    try:
        st.dataframe(result.df, use_container_width=True)
    except TypeError:
        st.dataframe(result.df)

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

    if result.chart_explanation:
        st.markdown("### What This Means (Plain English)")
        st.write(result.chart_explanation)

if st.session_state.history:
    st.subheader("Conversation Memory")
    for i, h in enumerate(reversed(st.session_state.history[-8:]), start=1):
        st.markdown(f"**{i}. Q:** {h['question']}")
        st.code(h["sql"], language="sql")
