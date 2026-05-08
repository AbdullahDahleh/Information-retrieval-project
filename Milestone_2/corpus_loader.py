
import json, os, pandas as pd

def load_corpus(corpus_path):
    ext = os.path.splitext(corpus_path)[1].lower()

    if ext == '.json':
        with open(corpus_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

    elif ext == '.csv':
        documents = pd.read_csv(corpus_path).to_dict('records')

    else:
        raise ValueError(f"Unsupported format: {ext}")

    normalized = []
    for i, doc in enumerate(documents):
        normalized.append({
            # Unique ID (generate if missing)
            'id': doc.get('id', f'doc_{i:04d}'),

            # Core fields
            'title': doc.get('title', 'Untitled'),
            'text': doc.get('text', ''),
            'topic': doc.get('topic', 'Unknown'),
            'link': doc.get('link', f'https://corpus.local/doc/{i}'),

            # Additional fields from your CSV
            'type': doc.get('type', ''),
            'doc_length': doc.get('doc_length', 0),
        })

    print(f"Loaded {len(normalized)} documents.")
    return normalized
