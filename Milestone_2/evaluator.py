
import numpy as np

class Evaluator:
    def precision_at_k(self, retrieved_ids, relevant_ids, k=10):
        hits = sum(1 for d in retrieved_ids[:k] if d in set(relevant_ids))
        return hits / k

    def recall(self, retrieved_ids, relevant_ids):
        if not relevant_ids: return 0.0
        return sum(1 for d in retrieved_ids if d in set(relevant_ids)) / len(relevant_ids)

    def average_precision(self, retrieved_ids, relevant_ids):
        relevant_set = set(relevant_ids)
        hits = 0; sum_p = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                hits += 1; sum_p += hits / (i + 1)
        return sum_p / len(relevant_ids) if relevant_ids else 0.0

    def evaluate_model(self, model_name, queries_results):
        p10s, recalls, aps = [], [], []
        for qr in queries_results:
            p10s.append(self.precision_at_k(qr['retrieved_ids'], qr['relevant_ids']))
            recalls.append(self.recall(qr['retrieved_ids'], qr['relevant_ids']))
            aps.append(self.average_precision(qr['retrieved_ids'], qr['relevant_ids']))
        return {
            'model': model_name,
            'avg_precision@10': np.mean(p10s),
            'avg_recall': np.mean(recalls),
            'MAP': np.mean(aps),
            'per_query_p10': p10s, 'per_query_recall': recalls, 'per_query_ap': aps,
        }
