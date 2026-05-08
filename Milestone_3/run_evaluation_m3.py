"""
run_evaluation_m3.py
Compares baseline M2 system vs LLM-augmented system on all 20 queries.
Outputs: evaluation_m3_results.csv + prints comparison table.
"""
import pandas as pd, os
from corpus_loader    import load_corpus
from query_handler    import QueryHandler
from models.vsm       import VSMRetriever
from models.bm25      import BM25Retriever
from models.embedding import EmbeddingRetriever
from evaluator        import Evaluator
from queries          import QUERIES
from llm_augment      import LLMAugmenter

# ── helpers ──────────────────────────────────────────────────────────────────
def get_relevant_ids(documents, relevant_topics):
    return [d["id"] for d in documents if d["topic"] in relevant_topics]

def run_model_on_queries(model, queries, documents, qh, llm=None):
    """Run all queries through a model. Optionally apply LLM rewriting first."""
    evaluator    = Evaluator()
    qr_list      = []
    rewrite_log  = []   # track original vs rewritten for qualitative analysis

    for q in queries:
        # Step 1: base preprocessing
        proc = qh.process_query(q["query"])
        search_q = proc["processed"]

        # Step 2: optional LLM rewriting
        rewritten = None
        if llm:
            rw_result = llm.rewrite_query(search_q)
            if rw_result["success"]:
                rewritten = rw_result["rewritten"]
                search_q  = rewritten

        rewrite_log.append({
            "query_id":  q["id"],
            "query_type": q["type"],
            "original":   q["query"],
            "processed":  proc["processed"],
            "rewritten":  rewritten if rewritten else proc["processed"],
        })

        # Step 3: retrieval
        results      = model.search(search_q, top_k=10, phrase=proc["phrase"])
        retrieved_ids = [r["doc"]["id"] for r in results]
        relevant_ids  = get_relevant_ids(documents, q["relevant_topics"])

        qr_list.append({
            "retrieved_ids": retrieved_ids,
            "relevant_ids":  relevant_ids,
        })

    metrics = evaluator.evaluate_model("model", qr_list)
    return metrics, rewrite_log

# ── main ─────────────────────────────────────────────────────────────────────
print("Loading corpus...")
docs = load_corpus("corpus/corpus.csv")
vsm   = VSMRetriever();       vsm.fit(docs)
bm25  = BM25Retriever();      bm25.fit(docs)
embed = EmbeddingRetriever(); embed.fit(docs)

qh  = QueryHandler()

print("\nLoading LLM augmenter...")
llm = LLMAugmenter()

model_dict = {"VSM": vsm, "BM25": bm25, "Embedding": embed}
rows       = []
all_logs   = []

for model_name, model in model_dict.items():
    print(f"\n── {model_name}: baseline ──")
    base_metrics, _ = run_model_on_queries(model, QUERIES, docs, qh, llm=None)

    print(f"── {model_name}: LLM-augmented ──")
    llm_metrics,  log = run_model_on_queries(model, QUERIES, docs, qh, llm=llm)
    for entry in log:
        entry["model"] = model_name
    all_logs.extend(log)

    rows.append({
        "Model":               model_name,
        "Mode":                "Baseline (M2)",
        "Precision@10":        round(base_metrics["avg_precision@10"], 4),
        "Recall":              round(base_metrics["avg_recall"],        4),
        "MAP":                 round(base_metrics["MAP"],               4),
    })
    rows.append({
        "Model":               model_name,
        "Mode":                "LLM-Augmented (M3)",
        "Precision@10":        round(llm_metrics["avg_precision@10"],   4),
        "Recall":              round(llm_metrics["avg_recall"],          4),
        "MAP":                 round(llm_metrics["MAP"],                 4),
    })

    delta_p   = llm_metrics["avg_precision@10"] - base_metrics["avg_precision@10"]
    delta_map = llm_metrics["MAP"]              - base_metrics["MAP"]
    print(f"  ΔPrecision@10: {delta_p:+.4f}   ΔMAP: {delta_map:+.4f}")

# ── print comparison table ────────────────────────────────────────────────────
df = pd.DataFrame(rows)
print("\n════════ FULL COMPARISON TABLE ════════")
print(df.to_string(index=False))
df.to_csv("evaluation_m3_results.csv", index=False)
print("\n✅ Saved → evaluation_m3_results.csv")

# ── print rewrite log ─────────────────────────────────────────────────────────
log_df = pd.DataFrame(all_logs)[["model","query_id","query_type","original","rewritten"]].drop_duplicates(
    subset=["query_id","query_type","original"])
log_df.to_csv("rewrite_log.csv", index=False)
print("✅ Saved → rewrite_log.csv")

# ── print a few example rewrites ─────────────────────────────────────────────
print("\n════ EXAMPLE QUERY REWRITES ════")
sample = log_df[log_df["model"]=="BM25"].head(10)
for _, row in sample.iterrows():
    print(f"\n[{row['query_id']} | {row['query_type']}]")
    print(f"  Original : {row['original']}")
    print(f"  Rewritten: {row['rewritten']}")
