
import streamlit as st, time
from corpus_loader    import load_corpus
from query_handler    import QueryHandler
from models.vsm       import VSMRetriever
from models.bm25      import BM25Retriever
from models.embedding import EmbeddingRetriever

st.set_page_config(page_title="IR System", page_icon="🔍", layout="wide")

@st.cache_resource(show_spinner=False)
def load_and_index():
    docs  = load_corpus('corpus/corpus.csv')
    vsm   = VSMRetriever();       vsm.fit(docs)
    bm25  = BM25Retriever();      bm25.fit(docs)
    embed = EmbeddingRetriever(); embed.fit(docs)
    return docs, vsm, bm25, embed

def get_snippet(content, query, max_chars=200):
    idx = content.lower().find(query.lower().split()[0])
    if idx == -1: return content[:max_chars] + ' ...'
    start   = max(0, idx - 40)
    snippet = content[start: start + max_chars]
    return ('... ' if start > 0 else '') + snippet + ' ...'

def main():
    st.title("🔍 Information Retrieval System")
    st.caption("500 documents · 8 topics · 3 retrieval models")

    with st.spinner("⏳ Loading corpus and indexing (first run may take ~2-4 min for embeddings)..."):
        docs, vsm, bm25, embed = load_and_index()
    st.success(f"✅ Ready — {len(docs)} documents indexed")

    qh = QueryHandler()

    with st.sidebar:
        st.header("⚙️ Settings")
        model_name = st.selectbox("Retrieval Model",
            ["Vector Space Model (VSM)", "BM25", "Embedding-based"])
        top_k = st.slider("Results to show", 5, 20, 10)
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

    query = st.text_input("🔎 Search Query",
        placeholder='Try:  AI security   |   "deep learning"   |   mchine lerning')
    st.caption('Wrap phrases in quotes for exact matching, e.g. `"neural network"`')

    if not query.strip():
        return

    processed = qh.process_query(query)

    with st.expander("🔧 Query Processing Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Original:** `{processed['original']}`")
        col2.markdown(f"**Processed:** `{processed['processed']}`")
        if processed['corrections']:
            fixes = ', '.join(f"`{k}` → `{v}`" for k,v in processed['corrections'].items())
            col3.markdown(f"**Spelling fixed:** {fixes}")
        if processed['phrase']:
            st.info(f'📌 Exact phrase mode: **"{processed["phrase"]}"**')

    with st.spinner(f"Searching with {model_name}..."):
        t0 = time.time()
        if   "VSM"   in model_name: results = vsm.search(processed['processed'],   top_k, processed['phrase'])
        elif "BM25"  in model_name: results = bm25.search(processed['processed'],  top_k, processed['phrase'])
        else:                       results = embed.search(processed['processed'],  top_k, processed['phrase'])
        elapsed = time.time() - t0

    st.markdown(f"### 📋 Top {len(results)} Results &nbsp; <small>({elapsed:.3f}s)</small>", unsafe_allow_html=True)

    if not results:
        st.warning("No results found. Try different keywords.")
        return

    for res in results:
        doc = res['doc']
        with st.container():
            left, right = st.columns([6, 1])
            with left:
                st.markdown(f"**#{res['rank']}. [{doc['title']}]({doc['link']})**")
                st.caption(f"📁 {doc['topic']}  ·  🔗 {doc['link']}")
                st.markdown(f"<small style='color:gray'>{get_snippet(doc['text'], processed['processed'])}</small>",
                            unsafe_allow_html=True)
            with right:
                st.metric("Score", f"{res['score']:.4f}")
            st.divider()

if __name__ == "__main__":
    main()
