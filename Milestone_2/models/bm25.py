
import numpy as np, re, nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1=k1; self.b=b; self.bm25=None; self.documents=None

    def _tokenize(self, text):
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        return word_tokenize(text)

    def fit(self, documents):
        self.documents = documents
        tokenized = [
        self._tokenize(d['title'] + ' ' + d['text']) for d in documents]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        print(f"BM25: Indexed {len(documents)} documents.")

    def search(self, query, top_k=10, phrase=None):
        scores = self.bm25.get_scores(self._tokenize(query))
        if phrase:
            for i, doc in enumerate(self.documents):
                if phrase.lower() in (doc['title'] + ' ' + doc['text']).lower():
                      scores[i] *= 1.5
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{'doc': self.documents[i], 'score': float(scores[i]), 'rank': r+1}
                for r, i in enumerate(top_indices) if scores[i] > 0]
