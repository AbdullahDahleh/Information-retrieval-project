"""
llm_augment.py
Two LLM augmentation strategies for the IR pipeline:
  1. Query Rewriting  — sends raw query to LLM, returns improved query
  2. Result Summarization — sends top-10 docs to LLM, returns a summary
"""
import os
import time
from openai import OpenAI

class LLMAugmenter:

    def __init__(self):
        api_key  = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

        if not api_key:
            raise ValueError(
                "LLM_API_KEY environment variable not set.\n"
                "Run: os.environ['LLM_API_KEY'] = 'your_key'"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._cache = {}   # avoid calling LLM twice for the same input
        print(f"LLMAugmenter ready — model: {self.model}")

    # ── internal helper ─────────────────────────────────────────────────────
    def _call(self, system_prompt, user_prompt, max_tokens=300, retries=2):
        key = hash(system_prompt + user_prompt)
        if key in self._cache:
            return self._cache[key]

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,   # low temp = deterministic, factual
                )
                result = resp.choices[0].message.content.strip()
                self._cache[key] = result
                return result
            except Exception as e:
                print(f"LLM call attempt {attempt+1} failed: {e}")
                time.sleep(1)

        return None   # both attempts failed


    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 1 — QUERY REWRITING
    # Send the raw user query → get a cleaner, richer retrieval query back
    # ══════════════════════════════════════════════════════════════════════
    def rewrite_query(self, original_query: str) -> dict:
        """
        Returns a dict:
          {
            'rewritten':    str,   # the improved query to pass to retrieval
            'changes':      str,   # short explanation of what was changed
            'success':      bool
          }
        """
        system = (
            "You are an expert Information Retrieval system. "
            "Your task is to rewrite a user search query to make it more effective "
            "for retrieving technical documents. Follow these rules strictly:\n"
            "1. Fix all spelling errors\n"
            "2. Expand abbreviations (AI→Artificial Intelligence, IoT→Internet of Things, etc.)\n"
            "3. Add 2-3 key technical synonyms or related terms that would appear in relevant documents\n"
            "4. Remove vague or stop-word-heavy phrasing\n"
            "5. Keep the rewritten query under 20 words\n"
            "6. Return EXACTLY two lines:\n"
            "   REWRITTEN: <the rewritten query>\n"
            "   CHANGES: <one short sentence explaining what you changed>\n"
            "Do not add any other text."
        )
        user = f"Original query: {original_query}"

        response = self._call(system, user, max_tokens=120)

        if response and "REWRITTEN:" in response:
            lines  = response.strip().split("\n")
            rw     = next((l.replace("REWRITTEN:", "").strip() for l in lines if l.startswith("REWRITTEN:")), original_query)
            chng   = next((l.replace("CHANGES:", "").strip()   for l in lines if l.startswith("CHANGES:")),   "No changes noted.")
            return {"rewritten": rw, "changes": chng, "success": True}

        # fallback — LLM failed or returned unexpected format
        return {"rewritten": original_query, "changes": "LLM unavailable, using original query.", "success": False}


    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY 2 — RESULT SUMMARIZATION
    # Send top-10 retrieved docs → get a coherent summary paragraph back
    # ══════════════════════════════════════════════════════════════════════
    def summarize_results(self, query: str, results: list) -> dict:
        """
        results: list of dicts from retrieval model, each with 'doc' key
        Returns:
          {
            'summary':  str,    # 3-4 sentence summary of retrieved content
            'themes':   list,   # 2-4 main themes detected
            'success':  bool
          }
        """
        if not results:
            return {"summary": "No results to summarize.", "themes": [], "success": False}

        # Build compact context from top-5 docs (top-10 might exceed token limit)
        doc_lines = []
        for i, r in enumerate(results[:5], 1):
            title   = r["doc"].get("title",   "Untitled")
            snippet = r["doc"].get("content", "")[:250].replace("\n", " ")
            doc_lines.append(f"[{i}] {title}: {snippet}")
        context = "\n".join(doc_lines)

        system = (
            "You are a research assistant helping a user understand search results. "
            "Given a search query and snippets from retrieved documents, do two things:\n"
            "1. Write a 3-4 sentence paragraph summarizing what the documents collectively cover "
            "and how they relate to the query.\n"
            "2. List the 2-4 main technical themes present across the results.\n"
            "Return EXACTLY in this format:\n"
            "SUMMARY: <your paragraph here>\n"
            "THEMES: <comma-separated theme list>\n"
            "No other text."
        )
        user = (
            f"Search query: {query}\n\n"
            f"Retrieved documents:\n{context}\n\n"
            "Generate the summary and themes."
        )

        response = self._call(system, user, max_tokens=350)

        if response and "SUMMARY:" in response:
            lines   = response.strip().split("\n")
            summary = next((l.replace("SUMMARY:", "").strip() for l in lines if l.startswith("SUMMARY:")), "")
            themes_raw = next((l.replace("THEMES:", "").strip() for l in lines if l.startswith("THEMES:")), "")
            themes  = [t.strip() for t in themes_raw.split(",") if t.strip()]
            return {"summary": summary, "themes": themes, "success": True}

        return {"summary": "Summary unavailable — LLM call failed.", "themes": [], "success": False}
