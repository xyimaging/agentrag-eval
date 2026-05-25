"""
BaseRAGPipeline — shared retrieval, LLM routing, and cost tracking.

All three pipeline variants (Naive, Self-Reflective, Agentic) inherit
from this class so that retrieval quality and LLM behaviour are held
constant across comparisons.

Supports both OpenAI and Anthropic via LLM_PROVIDER env var.
"""

import os
import time
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    TOP_K_RETRIEVE, EMBEDDING_MODEL,
)


class BaseRAGPipeline:
    """
    Provider-agnostic base class. Set LLM_PROVIDER=openai|anthropic in .env.
    Subclasses only need to implement answer().
    """

    def __init__(self, collection, name: str = "BaseRAG"):
        self.name = name
        self.collection = collection
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.provider = LLM_PROVIDER
        self.total_tokens = 0
        self.total_api_calls = 0

        if self.provider == "anthropic":
            import anthropic as anthropic_sdk
            self._anthropic = anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_MODEL
            self.client = None
        else:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
            self._anthropic = None

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVE) -> list[dict]:
        """Embed query, return top-k passages as [{title, text, score}]."""
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        passages = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            passages.append({
                "title": meta.get("title", ""),
                "text":  doc,
                "score": 1 - dist,  # cosine distance → similarity
            })
        return passages

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def call_llm(self, messages: list[dict], temperature: float = 0.0) -> str:
        self.total_api_calls += 1
        if self.provider == "anthropic":
            return self._call_anthropic(messages, temperature)
        return self._call_openai(messages, temperature)

    def _call_openai(self, messages: list[dict], temperature: float) -> str:
        for attempt in range(6):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                self.total_tokens += response.usage.total_tokens
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    time.sleep(2 ** attempt)  # exponential back-off: 1s→2s→4s…
                else:
                    raise
        raise RuntimeError("OpenAI rate limit: max retries exceeded")

    def _call_anthropic(self, messages: list[dict], temperature: float) -> str:
        system_msg = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_msgs.append(m)
        kwargs = dict(model=self.model, max_tokens=512,
                      temperature=temperature, messages=user_msgs)
        if system_msg:
            kwargs["system"] = system_msg
        response = self._anthropic.messages.create(**kwargs)
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text.strip()

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def format_passages(passages: list[dict]) -> str:
        lines = []
        for i, p in enumerate(passages, 1):
            lines.append(f"[Passage {i}] {p['title']}\n{p['text']}")
        return "\n\n".join(lines)

    def answer(self, question: str) -> dict:
        """
        Must return a dict with at minimum:
          answer: str, api_calls: int, retrieved_passages: list
        """
        raise NotImplementedError

    def reset_stats(self):
        self.total_tokens = 0
        self.total_api_calls = 0

    def cost_estimate_usd(self) -> float:
        """Rough cost estimate at ~$0.40/1M tokens (conservative blended rate)."""
        return self.total_tokens / 1_000_000 * 0.40
