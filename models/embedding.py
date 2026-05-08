
import numpy as np, pickle, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name=model_name; self.model=None
        self.embeddings=None; self.documents=None

    def fit(self, documents, cache_path='embeddings_cache.pkl'):
        self.documents = documents
        if os.path.exists(cache_path):
            print("Embeddings: Loading from cache...")
            with open(cache_path,'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            print("Embeddings: Computing (first time, ~2-4 min)...")
            self.model = SentenceTransformer(self.model_name)
            texts = [d['title'] + ' ' + d['text'][:512] for d in documents]
            self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            with open(cache_path,'wb') as f:
                pickle.dump(self.embeddings, f)
            print("Saved to cache.")
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        print(f"Embeddings: Ready. ({len(documents)} docs)")

    def search(self, query, top_k=10, phrase=None):
        query_emb = self.model.encode([query])
        scores = cosine_similarity(query_emb, self.embeddings).flatten()
        if phrase:
            for i, doc in enumerate(self.documents):
                if phrase.lower() in (doc['title'] + ' ' + doc['text']).lower():
                    scores[i] *= 1.3
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{'doc': self.documents[i], 'score': float(scores[i]), 'rank': r+1}
                for r, i in enumerate(top_indices)]
