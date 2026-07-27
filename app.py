from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st
import truststore
from openai import OpenAI


truststore.inject_into_ssl()

APP_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = APP_DIR.parent


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overwriting existing variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def get_setting(name: str, default: str | Path | None = None) -> str:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    try:
        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except (FileNotFoundError, KeyError):
        pass
    return str(default) if default is not None else ""


load_env_file(APP_DIR / ".ENV")
load_env_file(WORKSPACE_DIR / ".ENV")

KNOWLEDGE_BASE_DIR = Path(
    get_setting("GCC_KNOWLEDGE_BASE_DIR", APP_DIR / "knowledge_base")
)
ROOT_INDEX = KNOWLEDGE_BASE_DIR / "index.md"
DEFAULT_MODEL = get_setting("OPENAI_MODEL", "gpt-5.6-luna")


@st.cache_data(show_spinner=False)
def load_indexes() -> tuple[str, dict[str, str]]:
    root_text = ROOT_INDEX.read_text(encoding="utf-8")
    group_indexes: dict[str, str] = {}
    for directory in sorted(KNOWLEDGE_BASE_DIR.iterdir()):
        index_path = directory / "index.md"
        if directory.is_dir() and index_path.exists():
            group_indexes[directory.name] = index_path.read_text(encoding="utf-8")
    return root_text, group_indexes


@st.cache_data(show_spinner=False)
def contract_catalog() -> dict[str, Path]:
    return {
        path.name: path
        for path in KNOWLEDGE_BASE_DIR.rglob("*.md")
        if path.name != "index.md"
    }


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def local_fallback_documents(question: str, limit: int = 3) -> list[str]:
    """Small lexical fallback used only if the routing response is invalid."""
    stopwords = {
        "about",
        "after",
        "agreement",
        "contract",
        "could",
        "does",
        "from",
        "have",
        "into",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "there",
        "their",
    }
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) > 3 and token not in stopwords
    }
    scored: list[tuple[int, str]] = []
    for filename, path in contract_catalog().items():
        text = path.read_text(encoding="utf-8").lower()
        score = sum(text.count(token) for token in tokens)
        scored.append((score, filename))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [name for score, name in scored if score > 0][:limit]
    return selected or [sorted(contract_catalog())[0]]


def route_question(client: OpenAI, question: str, model: str) -> dict[str, Any]:
    root_index, group_indexes = load_indexes()
    index_bundle = "\n\n".join(
        f"--- GROUP INDEX: {folder}/index.md ---\n{text}"
        for folder, text in group_indexes.items()
    )
    available_files = sorted(contract_catalog())

    prompt = f"""
You are the retrieval router for a synthetic India GCC contract knowledge base.

Your job is routing only. Do not answer the user's legal or commercial question.
Follow the root index first, then the five group indexes. Select the smallest
useful set of sources: normally one category and one contract, with a maximum
of two categories and three contracts.

Return JSON only, using exactly this shape:
{{
  "selected_categories": ["exact category folder"],
  "selected_documents": ["exact markdown filename"],
  "rationale": "one short sentence",
  "search_terms": ["term 1", "term 2"]
}}

Only select filenames from this list:
{json.dumps(available_files, indent=2)}

USER QUESTION:
{question}

ROOT INDEX:
{root_index}

GROUP INDEXES:
{index_bundle}
""".strip()

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=500,
    )
    route = parse_json_object(extract_output_text(response))
    catalog = contract_catalog()

    selected_documents = route.get("selected_documents", [])
    if not isinstance(selected_documents, list):
        selected_documents = []
    selected_documents = [
        str(filename)
        for filename in selected_documents
        if str(filename) in catalog
    ][:3]

    if not selected_documents:
        selected_documents = local_fallback_documents(question)
        route["rationale"] = (
            "The model router returned no valid file, so the local lexical "
            "fallback selected the closest documents."
        )

    selected_categories = route.get("selected_categories", [])
    if not isinstance(selected_categories, list):
        selected_categories = []
    selected_categories = [str(value) for value in selected_categories][:2]

    if not selected_categories:
        selected_categories = sorted(
            {
                catalog[filename].parent.name
                for filename in selected_documents
            }
        )

    route["selected_categories"] = selected_categories
    route["selected_documents"] = selected_documents
    route["search_terms"] = route.get("search_terms", [])
    return route


def build_source_bundle(selected_documents: list[str]) -> tuple[str, list[dict[str, str]]]:
    catalog = contract_catalog()
    source_blocks: list[str] = []
    sources: list[dict[str, str]] = []

    for filename in selected_documents:
        path = catalog[filename]
        relative = path.relative_to(KNOWLEDGE_BASE_DIR).as_posix()
        text = path.read_text(encoding="utf-8")
        document_id_match = re.search(r"(?m)^document_id:\s*(.+)$", text)
        title_match = re.search(r"(?m)^title:\s*(.+)$", text)
        document_id = (
            document_id_match.group(1).strip()
            if document_id_match
            else filename
        )
        title = title_match.group(1).strip() if title_match else filename

        sources.append(
            {
                "document_id": document_id,
                "title": title,
                "path": relative,
            }
        )
        source_blocks.append(
            f"--- SOURCE {document_id}: {title} ({relative}) ---\n{text}"
        )

    return "\n\n".join(source_blocks), sources


def answer_question(
    client: OpenAI,
    question: str,
    route: dict[str, Any],
    chat_history: list[dict[str, Any]],
    model: str,
) -> tuple[str, list[dict[str, str]]]:
    source_bundle, sources = build_source_bundle(route["selected_documents"])
    recent_history = [
        {"role": item["role"], "content": item["content"]}
        for item in chat_history[-6:]
        if item.get("role") in {"user", "assistant"}
    ]

    prompt = f"""
You are a contract knowledge-base assistant for a synthetic India GCC demo.
Answer the current question only from the supplied contract sources.

Rules:
- State the answer directly and concisely.
- Preserve exact amounts, dates, locations, thresholds, exceptions and Party roles.
- Cite every material contract-specific claim inline in the format
  [DOCUMENT_ID — Exact Section Heading].
- Never invent a clause or use general legal knowledge to fill a gap.
- If the sources do not answer the question, say that clearly.
- If sources conflict, describe the difference rather than choosing silently.
- Finish with a short "Sources consulted" list.
- Remind the user only when relevant that the dataset is synthetic.

RETRIEVAL ROUTE:
{json.dumps(route, indent=2)}

RECENT CONVERSATION:
{json.dumps(recent_history, indent=2)}

CURRENT QUESTION:
{question}

CONTRACT SOURCES:
{source_bundle}
""".strip()

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=1400,
    )
    return extract_output_text(response).strip(), sources


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask me about the 25 synthetic India GCC contracts. "
                    "I will route through the knowledge-base indexes, retrieve "
                    "the relevant agreements, and show the sources I used."
                ),
                "route": None,
                "sources": [],
            }
        ]


def render_route(route: dict[str, Any] | None, sources: list[dict[str, str]]) -> None:
    if not route:
        return
    with st.expander("Retrieval trace", expanded=False):
        st.markdown("**Categories**")
        for category in route.get("selected_categories", []):
            st.code(category, language=None)

        st.markdown("**Selected contracts**")
        for source in sources:
            st.markdown(
                f"- `{source['document_id']}` — {source['title']}\n"
                f"  - `{source['path']}`"
            )

        if route.get("rationale"):
            st.markdown(f"**Why:** {route['rationale']}")
        if route.get("search_terms"):
            st.markdown(
                "**Search terms:** "
                + ", ".join(f"`{term}`" for term in route["search_terms"])
            )


st.set_page_config(
    page_title="GCC Contract Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1000px; padding-top: 2rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.2); }
      .kb-badge {
        display:inline-block; padding:.2rem .55rem; border-radius:999px;
        background:#eef2ff; color:#3730a3; font-size:.78rem; font-weight:600;
        margin-bottom:.45rem;
      }
      .subtle { color:#6b7280; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_state()

with st.sidebar:
    st.title("GCC Knowledge Base")
    st.caption("25 synthetic contracts · 5 lifecycle groups")

    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.caption("The API key is loaded from an environment variable or Streamlit Secrets.")

    st.divider()
    st.subheader("Try a question")
    examples = [
        "What replacement guarantee applies to recruitment hires?",
        "Compare managed workspace and office lease obligations.",
        "Who can approve or release payments under the finance agreement?",
        "What must be transferred when a BOT arrangement exits?",
        "When are three vendor quotations required?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

    st.divider()
    st.caption(
        "Retrieval path: root index → category index → selected contracts → answer."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

st.markdown('<div class="kb-badge">INDEX-FIRST RAG DEMO</div>', unsafe_allow_html=True)
st.title("GCC Contract Knowledge Assistant")
st.markdown(
    '<div class="subtle">Grounded answers with visible category and document routing.</div>',
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_route(message.get("route"), message.get("sources", []))

question = st.chat_input("Ask about the GCC contract knowledge base…")
if st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "route": None, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    api_key = get_setting("OPENAI_API_KEY")
    if not api_key:
        error = (
            "OPENAI_API_KEY was not found. Configure it as an environment "
            "variable, local `.ENV` value, or Streamlit Secret."
        )
        with st.chat_message("assistant"):
            st.error(error)
        st.session_state.messages.append(
            {"role": "assistant", "content": error, "route": None, "sources": []}
        )
    elif not ROOT_INDEX.exists():
        error = f"Knowledge-base index not found: `{ROOT_INDEX}`"
        with st.chat_message("assistant"):
            st.error(error)
        st.session_state.messages.append(
            {"role": "assistant", "content": error, "route": None, "sources": []}
        )
    else:
        client = OpenAI(api_key=api_key)
        try:
            with st.chat_message("assistant"):
                with st.status("Routing through the knowledge base…", expanded=True) as status:
                    st.write("Reading the root and category indexes")
                    route = route_question(client, question, model)
                    st.write(
                        "Selected: "
                        + ", ".join(route.get("selected_documents", []))
                    )
                    st.write("Reading the selected contracts")
                    answer, sources = answer_question(
                        client,
                        question,
                        route,
                        st.session_state.messages,
                        model,
                    )
                    status.update(label="Answer grounded in retrieved contracts", state="complete")

                st.markdown(answer)
                render_route(route, sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "route": route,
                    "sources": sources,
                }
            )
        except Exception as exc:
            message = f"Request failed: {exc}"
            with st.chat_message("assistant"):
                st.error(message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": message,
                    "route": None,
                    "sources": [],
                }
            )
