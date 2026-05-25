"""
Naive RAG Pipeline — single-pass retrieve-then-read baseline.

One retrieval step, one LLM call, no iteration. This is the primary
baseline that all more complex pipelines are measured against.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.base import BaseRAGPipeline
from config import TOP_K_RETRIEVE


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the question using ONLY the provided passages.
If the answer is not in the passages, say "I don't know".
Give a SHORT, DIRECT answer — typically 1-5 words. Do not explain."""

ANSWER_TEMPLATE = """Passages:
{passages}

Question: {question}

Answer (short and direct):"""


class NaiveRAGPipeline(BaseRAGPipeline):
    """Single retrieve → single generate. Exactly 1 API call per question."""

    def __init__(self, collection, k: int = TOP_K_RETRIEVE):
        super().__init__(collection, name="NaiveRAG")
        self.k = k

    def answer(self, question: str) -> dict:
        call_count_before = self.total_api_calls

        passages = self.retrieve(question, k=self.k)

        passages_text = self.format_passages(passages)
        prompt = ANSWER_TEMPLATE.format(passages=passages_text, question=question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        answer = self.call_llm(messages)

        return {
            "answer":             answer,
            "api_calls":          self.total_api_calls - call_count_before,
            "retrieved_passages": passages,
            "steps":              ["retrieve", "answer"],
        }
