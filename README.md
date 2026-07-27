# GCC Contract Knowledge Assistant

A small local Streamlit application demonstrating index-first retrieval over the
synthetic GCC contract knowledge base.

## Retrieval flow

1. Read the root `index.md`.
2. Read the five category indexes.
3. Ask the model to select the smallest relevant set of contract files.
4. Load only those contracts.
5. Generate an answer with inline document and section citations.

No vector database or ingestion step is required.

The app uses Python's `truststore` package so HTTPS certificate validation
follows the Windows system trust store. TLS verification remains enabled.

## Configuration

The app loads `OPENAI_API_KEY` from an uncommitted `.ENV` file for local use
or from Streamlit Secrets when deployed:

```text
gcc_rag_demo/
├── app.py
├── requirements.txt
├── README.md
└── knowledge_base/
    ├── index.md
    └── five category folders/
```

For local use, place an uncommitted `.ENV` file beside `app.py` or in its
parent directory:

```dotenv
OPENAI_API_KEY=your-key
```

Optional settings:

```dotenv
OPENAI_MODEL=gpt-5.6-luna
GCC_KNOWLEDGE_BASE_DIR=D:\optional\alternate\knowledge_base
```

Do not commit `.ENV` or an API key to source control.

## Install and run

From the `gcc_rag_demo` directory:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The browser normally opens automatically. If it does not, use the local URL
shown in the terminal.

## Deploy on Streamlit Community Cloud

1. Push this folder as a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the entrypoint to `app.py`.
4. Under the app's **Secrets** settings, add:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-5.6-luna"
```

5. Deploy. The bundled `knowledge_base/` directory is read directly from the
   repository; no external storage or ingestion job is required.

## Suggested demo questions

- What replacement guarantee applies to recruitment hires?
- Compare managed workspace and office lease obligations.
- Who can approve or release payments under the finance agreement?
- What must be transferred when a BOT arrangement exits?
- When are three vendor quotations required?

## Scope

All contracts are synthetic and provided only for retrieval and demonstration
purposes. The application does not provide legal advice.
