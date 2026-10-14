import os
import re
import time
import random
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------- CONFIG ----------

# Tried in order. A slow, overloaded, retired, or out-of-quota model is skipped fast.
# Check ai.google.dev/gemini-api/docs/deprecations from time to time.
MODELS = [
    "gemini-3.5-flash-lite",  # fastest, primary
    "gemini-3.8-flash",       # best quality backup
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
]

REQUEST_TIMEOUT = 12         # seconds before giving up on a slow model
RETRIES_PER_MODEL = 1        # one quick retry for transient errors only
ROUNDS = 2                   # full passes over all models before giving up
FULL_DOC_CHAR_LIMIT = 12000  # smaller docs are sent whole; larger use retrieval
CHUNKS_PER_QUESTION = 3

st.set_page_config(page_title="Document Q&A Bot", page_icon="📄", layout="wide")
st.title("📄 Document Q&A Bot")
st.caption("Upload a PDF and ask questions about its contents.")


# ---------- API KEYS ----------

def get_api_keys():
    keys = [
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GOOGLE_API_KEY_2"),  # optional backup key from another project
    ]
    keys = [k for k in keys if k]
    if not keys:
        st.error("GOOGLE_API_KEY is missing. Add it to your .env file.")
        st.stop()
    return keys


# ---------- MODEL ----------

@st.cache_resource
def get_llm(model_name, api_key):
    base = dict(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        max_output_tokens=1500,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,           # we handle retries ourselves
    )
    try:
        # minimal reasoning = much faster answers for simple extraction
        llm = ChatGoogleGenerativeAI(thinking_level="minimal", **base)
        print(f"[init] {model_name}: thinking_level=minimal OK")
        return llm
    except Exception as e:
        # old library version: run `pip install -U langchain-google-genai`
        print(
            f"[init] {model_name}: thinking_level NOT supported -> "
            f"slow default ({str(e)[:100]})"
        )
        return ChatGoogleGenerativeAI(**base)


# ---------- PROMPT ----------

PROMPT_TEMPLATE = """You are a high-speed, precise Document Question-Answering Assistant. Your primary directive is to extract accurate answers from the provided context quickly, structured cleanly, and without fluff.

### RESPONSE FORMAT RULES:
1. Direct Answers First: Immediately answer the user's question(s) using bullet points or structured subheadings.
2. {format_rule}
3. Strict Grounding: Rely ONLY on clear facts directly mentioned in the provided context. If the answer is not explicitly stated in the context, state: "Not mentioned in the document." Do not extrapolate or guess.
4. Concise Style: Keep responses punchy, factual, and strictly under 250 words per question to minimize generation latency.

### CONTEXT:
<document>
{document}
</document>

### USER QUESTIONS:
{questions}
"""

SINGLE_RULE = (
    "There is only ONE question. Do NOT write any 'Question 1' or numbered "
    "heading. Start directly with the answer. For a simple factual lookup "
    "(a number, date, or name), answer in one short line."
)
MULTI_RULE = (
    "Multiple questions are asked. Address each under a clear, numbered heading "
    "matching the question order (e.g., \"### Question 1: [Short Summary]\")."
)


def build_prompt(questions, context):
    if len(questions) == 1:
        questions_text = questions[0]
        format_rule = SINGLE_RULE
    else:
        questions_text = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        format_rule = MULTI_RULE

    return PROMPT_TEMPLATE.format(
        format_rule=format_rule,
        document=context,
        questions=questions_text,
    )


# ---------- PDF PROCESSING ----------

def process_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name

    try:
        pages = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    chunks = []
    for doc in docs:
        text = doc.page_content.strip()
        if text:
            page = doc.metadata.get("page")
            if page is not None:
                text = f"[Page {page + 1}]\n{text}"
            chunks.append(text)
    return chunks


# ---------- RETRIEVAL ----------

STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "to", "me", "tell",
    "about", "and", "or", "for", "this", "that", "document", "pdf", "can",
    "you", "please", "give", "show", "find", "from", "with", "was", "were",
    "are", "how", "why", "when", "where", "who", "which", "explain",
    "describe", "summarize", "details", "information",
}


def split_questions(query):
    parts = re.split(r"(?<=\?)\s+|\n+", query.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [query.strip()]


def find_relevant_chunks(question, chunks, limit):
    words = [
        w for w in re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
        if w not in STOP_WORDS and len(w) > 1
    ]
    if not words:
        words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())

    scored = []
    for i, chunk in enumerate(chunks):
        lower = chunk.lower()
        score = sum(
            len(re.findall(r"\b" + re.escape(w) + r"\b", lower)) for w in words
        )
        if score > 0:
            scored.append((score, i))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in scored[:limit]]


def build_context(questions, chunks):
    full_text = "\n\n".join(chunks)
    if len(full_text) <= FULL_DOC_CHAR_LIMIT:
        return full_text

    picked, seen = [], set()
    for q in questions:
        for idx in find_relevant_chunks(q, chunks, CHUNKS_PER_QUESTION):
            if idx not in seen:
                seen.add(idx)
                picked.append(idx)

    if not picked:  # nothing matched: fall back to the first chunks
        picked = list(range(min(4, len(chunks))))

    picked.sort()
    return "\n\n".join(chunks[i] for i in picked)


# ---------- RESPONSE TEXT ----------

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------- FAST FAILOVER ----------

def classify_error(error):
    t = str(error).lower()
    if any(x in t for x in ("api key", "api_key", "permission", "403", "unauthenticated", "401")):
        return "fatal"
    if any(x in t for x in ("404", "not found", "not supported", "no longer available")):
        return "skip_model"      # retired or wrong model name
    if any(x in t for x in ("429", "resource_exhausted", "quota")):
        return "skip_model"      # quota is usually per model
    if any(x in t for x in ("timeout", "timed out", "deadline")):
        return "skip_model"      # too slow: don't wait again, go to the next model
    return "retry"               # 503, SSL, empty response, etc.


def generate_with_failover(prompt, placeholder):
    keys = get_api_keys()
    last_error = None

    for round_no in range(ROUNDS):
        for model_name in MODELS:
            for key in keys:
                llm = get_llm(model_name, key)

                for attempt in range(RETRIES_PER_MODEL + 1):
                    t0 = time.perf_counter()
                    try:
                        text = ""
                        first_token = None

                        for chunk in llm.stream(prompt):
                            piece = extract_text(chunk.content)
                            if piece and first_token is None:
                                first_token = time.perf_counter() - t0
                            text += piece
                            placeholder.markdown(text + "▌")

                        text = text.strip()
                        if not text:
                            raise RuntimeError("empty response")

                        placeholder.markdown(text)
                        total = time.perf_counter() - t0
                        print(
                            f"[ok] {model_name}: first word "
                            f"{(first_token or 0):.1f}s, total {total:.1f}s"
                        )
                        return text, model_name

                    except Exception as error:
                        last_error = error
                        kind = classify_error(error)
                        print(
                            f"[{kind}] {model_name} after {time.perf_counter() - t0:.1f}s: "
                            f"{str(error)[:200]}"
                        )

                        if kind == "fatal":
                            raise
                        if kind == "skip_model":
                            break
                        if attempt < RETRIES_PER_MODEL:
                            time.sleep(0.5 + random.random())

        if round_no < ROUNDS - 1:
            placeholder.info("Servers are busy, retrying...")
            time.sleep(2 + random.random())

    raise last_error or RuntimeError("All models failed")


def friendly_error(error):
    t = str(error).lower()
    if any(x in t for x in ("api key", "api_key", "permission", "403", "401")):
        return "There is a problem with your API key. Check GOOGLE_API_KEY in your .env file."
    if any(x in t for x in ("429", "resource_exhausted", "quota")):
        return "The API quota is used up on all models. Wait for it to reset or enable billing."
    if any(x in t for x in ("404", "not found", "no longer available")):
        return "No available model was found. Update the MODELS list in the code."
    return f"Could not get an answer after several attempts: {error}"


# ---------- SESSION STATE ----------

for key, value in {
    "messages": [],
    "chunks": [],
    "file_hash": None,
    "answer_cache": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- SIDEBAR ----------

with st.sidebar:
    st.header("Your Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- LOAD PDF ONLY WHEN IT CHANGES ----------

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.md5(pdf_bytes).hexdigest()

    if current_hash != st.session_state.file_hash:
        with st.spinner("Reading your PDF..."):
            try:
                chunks = process_pdf(pdf_bytes)
                st.session_state.chunks = chunks
                st.session_state.file_hash = current_hash
                st.session_state.messages = []
                st.session_state.answer_cache = {}

                if not chunks:
                    st.warning(
                        "No readable text was found. This PDF may contain scanned pages."
                    )
            except Exception as error:
                st.error(f"Could not process the PDF: {error}")
                st.stop()

    st.sidebar.success(f"Loaded: {uploaded_file.name}")


# ---------- DISPLAY HISTORY ----------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- CHAT ----------

if uploaded_file is None:
    st.info("Upload a PDF to begin.")

elif st.session_state.chunks:
    query = st.chat_input("Ask anything about your PDF (you can ask several questions)...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            start = time.perf_counter()
            placeholder = st.empty()
            cache_key = (st.session_state.file_hash, query.strip().lower())
            used_model = "cache"

            try:
                if cache_key in st.session_state.answer_cache:
                    answer = st.session_state.answer_cache[cache_key]
                    placeholder.markdown(answer)
                else:
                    questions = split_questions(query)
                    context = build_context(questions, st.session_state.chunks)
                    prompt = build_prompt(questions, context)

                    placeholder.markdown("⏳ Thinking...")
                    answer, used_model = generate_with_failover(prompt, placeholder)
                    st.session_state.answer_cache[cache_key] = answer

                st.caption(
                    f"{used_model}"
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                answer = friendly_error(error)
                placeholder.error(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )