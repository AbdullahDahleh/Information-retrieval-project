
import re
from spellchecker import SpellChecker

ABBREVIATIONS = {
    'ai': 'artificial intelligence', 'ml': 'machine learning',
    'dl': 'deep learning', 'nlp': 'natural language processing',
    'iot': 'internet of things', 'db': 'database',
    'os': 'operating system', 'api': 'application programming interface',
    'sql': 'structured query language', 'nn': 'neural network',
    'cv': 'computer vision', 'rl': 'reinforcement learning',
    'llm': 'large language model', 'gpu': 'graphics processing unit',
    'cpu': 'central processing unit', 'p2p': 'peer to peer',
    'vpn': 'virtual private network', 'ddos': 'distributed denial of service',
    'nft': 'non fungible token', 'defi': 'decentralized finance',
    'qc': 'quantum computing',
}

class QueryHandler:
    def __init__(self):
        self.spell = SpellChecker()

    def expand_abbreviations(self, query):
        words = query.split()
        expanded = []
        for word in words:
            clean = word.lower().strip("'\".,!?")
            expanded.append(ABBREVIATIONS[clean] if clean in ABBREVIATIONS else word)
        return ' '.join(expanded)

    def correct_spelling(self, query):
        phrase_pattern = re.compile(r'"[^"]+"')
        phrases = phrase_pattern.findall(query)
        clean_query = query
        for i, phrase in enumerate(phrases):
            clean_query = clean_query.replace(phrase, f'__PHRASE_{i}__')

        words = clean_query.split()
        corrected_words, corrections = [], {}
        for word in words:
            if word.startswith('__PHRASE_') or word.lower() in ABBREVIATIONS or len(word) <= 2:
                corrected_words.append(word)
                continue
            correction = self.spell.correction(word)
            if correction and correction != word.lower():
                corrections[word] = correction
                corrected_words.append(correction)
            else:
                corrected_words.append(word)

        corrected_query = ' '.join(corrected_words)
        for i, phrase in enumerate(phrases):
            corrected_query = corrected_query.replace(f'__PHRASE_{i}__', phrase)
        return corrected_query, corrections

    def extract_phrase(self, query):
        match = re.search(r'"([^"]+)"', query)
        return match.group(1) if match else None

    def process_query(self, query):
        query_lower = query.lower()
        corrected, corrections = self.correct_spelling(query_lower)
        expanded = self.expand_abbreviations(corrected)
        phrase = self.extract_phrase(expanded)
        return {
            'original': query, 'processed': expanded,
            'corrections': corrections, 'phrase': phrase, 'expanded': expanded,
        }
