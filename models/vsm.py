
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VSMRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
        self.tfidf_matrix = None
        self.documents = None

    def fit(self, documents):
        self.documents = documents
        corpus = [d['title'] + ' ' + d['text'] for d in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print(f"VSM: Indexed {len(documents)} documents.")

    def search(self, query, top_k=10, phrase=None):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        if phrase:
            for i, doc in enumerate(self.documents):
                if phrase.lower() in (doc['title']+' '+doc['content']).lower():
                    scores[i] *= 1.5
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{'doc': self.documents[i], 'score': float(scores[i]), 'rank': r+1}
                for r, i in enumerate(top_indices) if scores[i] > 0]
