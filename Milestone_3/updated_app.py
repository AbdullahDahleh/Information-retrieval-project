import streamlit as st
import time, os

from corpus_loader    import load_corpus
from query_handler    import QueryHandler
from models.vsm       import VSMRetriever
from models.bm25      import BM25Retriever
from models.embedding import EmbeddingRetriever

st.set_page_config(page_title="IR System — M3", page_icon="🔍", layout="wide")

# ── Load & index corpus (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_index():
    docs  = load_corpus("corpus/corpus.csv")
    vsm   = VSMRetriever();       vsm.fit(docs)
    bm25  = BM25Retriever();      bm25.fit(docs)
    embed = EmbeddingRetriever(); embed.fit(docs)
    return docs, vsm, bm25, embed

# ── Load LLM augmenter (only when key exists) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_llm():
    try:
        from llm_augment import LLMAugmenter
        return LLMAugmenter()
    except Exception as e:
        return None

# ── Snippet helper ────────────────────────────────────────────────────────────
def get_snippet(content, query, max_chars=200):
    idx = content.lower().find(query.lower().split()[0]) if query.split() else -1
    start   = max(0, idx - 40) if idx != -1 else 0
    snippet = content[start: start + max_chars]
    return ("... " if start > 0 else "") + snippet + " ..."

# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🔍 IR System — Milestone 3: LLM-Augmented Retrieval")
    st.caption("500 documents · 8 topics · 3 retrieval models · LLM Query Rewriting + Summarization")

    with st.spinner("⏳ Loading corpus and indexes..."):
        docs, vsm, bm25, embed = load_and_index()
    st.success(f"✅ {len(docs)} documents indexed")

    qh  = QueryHandler()
    llm = load_llm()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Retrieval Settings")
        model_name = st.selectbox("Retrieval Model",
            ["Vector Space Model (VSM)", "BM25", "Embedding-based"])
        top_k = st.slider("Results to show", 5, 20, 10)

        st.divider()
        st.header("🤖 LLM Augmentation")

        # API key input in sidebar (password field — key not visible)
        if not os.getenv("LLM_API_KEY"):
            api_key_input = st.text_input("API Key (Groq/OpenAI)",
                type="password",
                placeholder="Paste your API key here",
                help="Key is not stored. It only lives in this session.")
            if api_key_input:
                os.environ["LLM_API_KEY"] = api_key_input
                st.rerun()   # re-run to load LLM augmenter with new key
        else:
            st.success("🔑 API key loaded")

        use_rewriting      = st.checkbox("✏️ Query Rewriting",
            value=False, disabled=(llm is None),
            help="LLM rewrites your query before retrieval for better coverage.")
        use_summarization  = st.checkbox("📝 Result Summarization",
            value=False, disabled=(llm is None),
            help="LLM summarizes the top retrieved documents.")

        if llm is None:
            st.warning("Set your API key above to enable LLM features.")

        st.divider()
        st.header("📊 Corpus Topics")
        st.markdown("""
| Topic | Docs |
|---|---|
| Artificial Intelligence | 93 |
| Cloud Computing | 81 |
| Cybersecurity | 67 |
| Blockchain | 60 |
| Data Science | 59 |
| Quantum Computing | 59 |
| Internet of Things | 47 |
| Computer Networks | 46 |
        """)

    # ── Search Box ─────────────────────────────────────────────────────────────
    raw_query = st.text_input("🔎 Search Query",
        placeholder='Try:  AI security   |   "deep learning"   |   mchine lerning')
    st.caption('Wrap phrases in quotes for exact matching, e.g. `"neural network"`')

    if not raw_query.strip():
        return

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Base preprocessing (spell correction, abbreviations, phrase detect)
    # ══════════════════════════════════════════════════════════════════════════
    processed = qh.process_query(raw_query)

    with st.expander("🔧 Query Processing Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Original:** `{processed['original']}`")
        col2.markdown(f"**After spell-check + expansion:** `{processed['processed']}`")
        if processed["corrections"]:
            fixes = ", ".join(f"`{k}` → `{v}`" for k,v in processed["corrections"].items())
            col3.markdown(f"**Spelling fixed:** {fixes}")
        if processed["phrase"]:
            st.info(f'📌 Exact phrase mode: **"{processed["phrase"]}"**')

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 (Optional): LLM Query Rewriting
    # ══════════════════════════════════════════════════════════════════════════
    final_query = processed["processed"]
    rewrite_info = None

    if use_rewriting and llm:
        with st.spinner("🤖 LLM rewriting query..."):
            rewrite_info = llm.rewrite_query(processed["processed"])

        if rewrite_info["success"]:
            final_query = rewrite_info["rewritten"]

        with st.expander("✏️ LLM Query Rewriting", expanded=True):
            col_a, col_b = st.columns(2)
            col_a.markdown("**Before rewriting:**")
            col_a.code(processed["processed"])
            col_b.markdown("**After LLM rewriting:**")
            col_b.code(final_query)
            if rewrite_info:
                st.caption(f"📎 Changes: {rewrite_info['changes']}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Retrieval
    # ══════════════════════════════════════════════════════════════════════════
    phrase = processed["phrase"]

    with st.spinner(f"Searching with {model_name}..."):
        t0 = time.time()
        if   "VSM"   in model_name: results = vsm.search(final_query,   top_k, phrase)
        elif "BM25"  in model_name: results = bm25.search(final_query,  top_k, phrase)
        else:                       results = embed.search(final_query,  top_k, phrase)
        elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 (Optional): LLM Result Summarization
    # ══════════════════════════════════════════════════════════════════════════
    if use_summarization and llm and results:
        with st.spinner("🤖 LLM summarizing results..."):
            summary_info = llm.summarize_results(final_query, results)

        if summary_info["success"]:
            st.markdown("---")
            st.markdown("### 📝 AI Summary of Results")
            st.info(summary_info["summary"])
            if summary_info["themes"]:
                theme_badges = "  ".join([f"`{t}`" for t in summary_info["themes"]])
                st.markdown(f"**Main themes detected:** {theme_badges}")
            st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Display Results
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"### 📋 Top {len(results)} Results &nbsp; <small>({elapsed:.3f}s)</small>",
        unsafe_allow_html=True)

    if not results:
        st.warning("No results found. Try different keywords.")
        return

    for res in results:
        doc = res["doc"]
        with st.container():
            left, right = st.columns([6, 1])
            with left:
                st.markdown(f"**#{res['rank']}. [{doc['title']}]({doc['link']})**")
                st.caption(f"📁 {doc['topic']}  ·  🔗 {doc['link']}")
                st.markdown(
                    f"<small style='color:gray'>{get_snippet(doc['text'], final_query)}</small>",
                    unsafe_allow_html=True)
            with right:
                st.metric("Score", f"{res['score']:.4f}")
            st.divider()

if __name__ == "__main__":
    main()
