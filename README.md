# Live Conversational BI Agent

This project is now structured as a live-style BI app:

- users can upload their own `CSV`, `Parquet`, or `DuckDB` files
- large CSV uploads are split into row-based partitions during ingestion
- DuckDB stores the prepared dataset for interactive querying
- the app keeps the same plain-English to SQL to table/chart workflow

## What Changed

The original version was hard-wired to the Instacart schema. The app now supports two modes:

- `Use bundled Instacart data`
- `Upload my files`

When users upload large CSVs, the backend reads them in chunks and creates partition tables such as:

- `sales__part_0001`
- `sales__part_0002`
- `sales__part_0003`

It then creates a logical view like `sales` over those partitions, so the BI agent can query one clean table name while DuckDB handles the physical storage efficiently.

## Current Capabilities

- conversational BI over uploaded datasets
- schema discovery from uploaded files
- partitioned ingestion for big CSV files
- automatic column profiling for numeric, categorical, and temporal fields
- likely relationship detection across uploaded tables
- metadata queries like tables, columns, and row counts
- chart selection for common query shapes
- plain-English chart explanation
- provider fallback: `Gemini`, `OpenAI`, `Groq`, then built-in heuristics

## Supported Upload Formats

- `.csv`
- `.parquet`
- `.duckdb`
- `.db`

## How Partitioning Works

For CSV uploads, the app:

1. saves the uploaded file into a session runtime folder
2. reads it in chunks with pandas
3. writes each chunk into its own DuckDB table
4. creates a single view over all chunk tables

This keeps ingestion safer for large files and avoids trying to load the whole file into memory at once.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Live

The app is prepared for Streamlit-style or Procfile-based deployment.

Repository: `https://github.com/Mourya611/Hireathon_BI_Agent`

Current deployment branch: `codex/live-bi-upload-deploy`

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Select the branch `codex/live-bi-upload-deploy` or merge that branch into `main`.
4. Set the main file to `app.py`.
5. Add secrets in the app settings if you want LLM-backed SQL generation:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
6. Deploy.

### Render / Railway / Similar Platforms

- `Procfile` is included.
- The app listens on the platform-provided `PORT`.
- Start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

## Environment Variables

Use `.env` for local secrets. `.env.example` is only a template.

- `GEMINI_API_KEY=`
- `GEMINI_MODEL=gemini-2.0-flash`
- `OPENAI_API_KEY=`
- `OPENAI_MODEL=gpt-5-mini`
- `GROQ_API_KEY=`
- `GROQ_BASE_URL=https://api.groq.com/openai/v1`
- `GROQ_MODEL=openai/gpt-oss-120b`
- `LLM_TIMEOUT_SECONDS=10`

## Example Questions

- `list the tables in my upload`
- `show the columns in the sales table`
- `rows in each table`
- `what relationships did you detect`
- `show monthly revenue trend`
- `top customers by total sales`
- `top departments by reorder rate`
- `orders by hour of day`
- `show the first 20 rows from customers`

## Notes

- Instacart-specific heuristics still work when that schema is loaded.
- Generic uploaded datasets work best when an LLM key is configured, because joins and business logic can vary from one schema to another.
- The UI now shows detected relationships to help users understand how uploaded tables may connect.
- Runtime upload artifacts are written to `runtime/` and ignored by git.
