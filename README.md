# Conversational BI Agent for Instacart

A chat-first business intelligence assistant that answers plain-English questions over the Instacart dataset by generating SQL, running it in DuckDB, and returning tables plus chart recommendations.

## What This System Does

- Loads the 6 Instacart CSV files into DuckDB views.
- Supports multi-table analysis across orders, products, aisles, and departments.
- Converts natural-language prompts into read-only SQL.
- Executes SQL with retry logic when first-generation SQL fails.
- Returns results as tables and an auto-selected chart (`bar`, `line`, `pie`, `scatter`, `histogram`, or table).
- Maintains short conversation memory for follow-up questions.

## 1. Architectural Clarity: Why This System Is Structured This Way

### Layered design

- UI layer: [`app.py`](/c:/Users/Nani/OneDrive/Desktop/hire_P!/app.py)
- BI/logic layer: [`bi_agent.py`](/c:/Users/Nani/OneDrive/Desktop/hire_P!/bi_agent.py)
- Data layer: `data/*.csv` queried through DuckDB

This split keeps responsibilities clean:

- `app.py` handles user input, rendering, and Streamlit state.
- `bi_agent.py` handles schema context, SQL generation, safety checks, retries, and chart logic.
- DuckDB handles heavy joins and aggregations.

### Why DuckDB over pure pandas

- `order_products__prior.csv` is very large (~32M+ rows plus header line count context).
- SQL analytics on large CSV-backed views is faster and more memory-efficient than repeated pandas merges.
- Generated SQL is inspectable and easier to debug in BI workflows.

### Modeling choices

- A unified `order_products_all` view is created using `UNION ALL` over prior + train product-order tables.
- Join paths are explicit and stable:
  - `order_products_all.order_id -> orders.order_id`
  - `order_products_all.product_id -> products.product_id`
  - `products.aisle_id -> aisles.aisle_id`
  - `products.department_id -> departments.department_id`
- `days_since_prior_order` remains nullable to avoid distorting first-order behavior.

## 2. Data Awareness: What Was Checked Before Building

Yes, data was inspected before finalizing architecture.

### Observed columns

- `orders.csv`: `order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order`
- `order_products__prior.csv`: `order_id,product_id,add_to_cart_order,reordered`
- `order_products__train.csv`: `order_id,product_id,add_to_cart_order,reordered`
- `products.csv`: `product_id,product_name,aisle_id,department_id`
- `aisles.csv`: `aisle_id,aisle`
- `departments.csv`: `department_id,department`

### Observed local file line counts (includes header)

- `orders.csv`: `3,421,084`
- `order_products__prior.csv`: `32,434,490`
- `order_products__train.csv`: `1,384,618`
- `products.csv`: `49,689`
- `aisles.csv`: `135`
- `departments.csv`: `22`

### How this changed the implementation

- Chose DuckDB + SQL-first execution instead of dataframe-heavy joins.
- Added a combined `order_products_all` abstraction to simplify natural-language analytics.
- Kept null handling explicit for temporal metrics.

## 3. Failure Honesty: Where the System Breaks and Why

Known limitations:

- Ambiguous prompts:
  - Example: "show loyalty trend" does not define metric/grain clearly.
  - Effect: SQL may be valid but not match user intent.
- Complex follow-ups:
  - Basic follow-up handling exists, but multi-step constrained rewrites can fail.
- Chart selection is heuristic:
  - Useful defaults, but not always the best visualization for every query shape.
- LLM dependency for long-tail requests:
  - Without provider keys or with provider downtime, system falls back to deterministic heuristics.
- Expensive broad queries:
  - Very large unconstrained aggregations can still be slow.

## 4. AI Fluency: Controlled AI Usage, Not Prompt-and-Pray

AI is used with guardrails:

- Prompt includes explicit schema and join graph.
- SQL is validated as read-only (`SELECT`/`WITH`) before execution.
- Failed SQL triggers retry with error-aware regeneration.
- Provider fallback chain is explicit:
  - `Gemini -> OpenAI -> Groq -> built-in heuristics`
- Database execution is the source of truth; model output is not trusted blindly.

## 5. Professional Communication: Non-Technical Explanation

If explaining to a stakeholder:

"This tool lets analysts ask business questions in plain English instead of writing SQL manually. It safely translates those questions into queries, runs them on the Instacart data, and returns a table and chart to support decisions. It is designed for large datasets and can answer questions like reorder behavior, category performance, and shopping-time patterns."

## Setup

### Prerequisites

- Python 3.10+
- Pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run app

```bash
streamlit run app.py
```

## Dataset

This repository does not include the six large Instacart CSV files.

Download dataset from Kaggle:
- https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis

After downloading, place these files inside the local `data/` folder:
- `orders.csv`
- `order_products__prior.csv`
- `order_products__train.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

## Environment Variables

Use `.env` for local secrets. `.env.example` is a template only.

- `GEMINI_API_KEY=`
- `GEMINI_MODEL=gemini-2.0-flash`
- `OPENAI_API_KEY=`
- `OPENAI_MODEL=gpt-5-mini`
- `GROQ_API_KEY=`
- `GROQ_BASE_URL=https://api.groq.com/openai/v1`
- `GROQ_MODEL=openai/gpt-oss-120b`
- `LLM_TIMEOUT_SECONDS=10`

Load order in app:

1. `.env`
2. `.env.example` (fallback template)

## Security Note

- Never commit real API keys.
- If a key is exposed, revoke and rotate it immediately.

## Example Questions

- "Top departments by reorder rate"
- "Orders by hour of day"
- "Which aisles have the highest reorder rate and average basket position?"
- "Top products by times ordered and reorder rate"
- "Now filter that to only organic products"

## Next Improvements

- Better multi-turn SQL rewrite for complex follow-up context.
- Semantic metric layer for reusable business KPIs.
- Query cost and latency guardrails for very expensive prompts.
- Stronger visualization recommendation logic.
