
import pandas as pd
from corpus_loader    import load_corpus
from query_handler    import QueryHandler
from models.vsm       import VSMRetriever
from models.bm25      import BM25Retriever
from models.embedding import EmbeddingRetriever
from evaluator        import Evaluator
from queries          import QUERIES

def get_relevant_ids(documents, relevant_topics):
    return [d['id'] for d in documents if d['topic'] in relevant_topics]

docs  = load_corpus('corpus/corpus.csv')
vsm   = VSMRetriever();       vsm.fit(docs)
bm25  = BM25Retriever();      bm25.fit(docs)
embed = EmbeddingRetriever(); embed.fit(docs)

qh        = QueryHandler()
evaluator = Evaluator()
models    = {'VSM': vsm, 'BM25': bm25, 'Embedding': embed}
rows      = []

for model_name, model in models.items():
    print(f"\n── Evaluating {model_name} ──")
    qr_list = []
    for q in QUERIES:
        proc    = qh.process_query(q['query'])
        results = model.search(proc['processed'], top_k=10, phrase=proc['phrase'])
        qr_list.append({
            'retrieved_ids': [r['doc']['id'] for r in results],
            'relevant_ids':  get_relevant_ids(docs, q['relevant_topics']),
        })
    res = evaluator.evaluate_model(model_name, qr_list)
    print(f"  Precision@10 : {res['avg_precision@10']:.4f}")
    print(f"  Recall       : {res['avg_recall']:.4f}")
    print(f"  MAP          : {res['MAP']:.4f}")
    rows.append({'Model': model_name,
                 'Precision@10': round(res['avg_precision@10'],4),
                 'Recall':       round(res['avg_recall'],4),
                 'MAP':          round(res['MAP'],4)})

df = pd.DataFrame(rows)
print("\n════ FINAL COMPARISON ════")
print(df.to_string(index=False))
df.to_csv('evaluation_results.csv', index=False)
print("\n✅ Saved → evaluation_results.csv")
