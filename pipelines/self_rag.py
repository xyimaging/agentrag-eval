"""
Self-Reflective RAG Pipeline — iterative reflect-and-retrieve loop.

After an initial answer, the LLM reflects on its own confidence. If
uncertain, it issues a new retrieval query and refines the answer.
Repeats up to MAX_REFLECTION_ROUNDS total rounds.

Inspired by:
  - Self-RAG (Asai et al., ICLR 2024) — reflection tokens trained end-to-end
  - FLARE (Jiang et al., EMNLP 2023) — active retrieval triggered by uncertainty
  This implementation is prompt-based (no training required).
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.base import BaseRAGPipeline
from config import TOP_K_RETRIEVE, MAX_REFLECTION_ROUNDS


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer questions using ONLY the provided passages.
Give SHORT, DIRECT answers — typically 1-5 words."""

INITIAL_ANSWER_TEMPLATE = """Passages:
{passages}

Question: {question}

Provide a short, direct answer:"""

REFLECT_TEMPLATE = """You just answered a question. Now evaluate your confidence.

Question: {question}
Your answer: {answer}
Passages used: {passages_titles}

Is your answer confident and well-supported by the passages?
Reply with JSON only:
{{"confident": true/false, "reason": "one sentence", "new_query": "refined search query if not confident, else null"}}"""

REFINE_TEMPLATE = """Passages (updated):
{passages}

Question: {question}
Previous answer: {prev_answer}
Reason it was uncertain: {reason}

Provide a better, short, direct answer:"""


class SelfReflectiveRAGPipeline(BaseRAGPipeline):
    """Retrieve → answer → reflect → optionally re-retrieve, up to max_rounds."""

    def __init__(self, collection, k: int = TOP_K_RETRIEVE,
                 max_rounds: int = MAX_REFLECTION_ROUNDS):
        super().__init__(collection, name="SelfReflectiveRAG")
        self.k = k
        self.max_rounds = max_rounds

    def answer(self, question: str) -> dict:
        call_count_before = self.total_api_calls
        all_passages = []
        steps = []

        # Round 1: initial retrieve + answer
        passages = self.retrieve(question, k=self.k)
        all_passages.extend(passages)
        steps.append("retrieve_r1")

        passages_text = self.format_passages(passages)
        current_answer = self.call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": INITIAL_ANSWER_TEMPLATE.format(
                passages=passages_text, question=question)},
        ])
        steps.append("answer_r1")

        # Reflection loop (up to max_rounds - 1 additional retrieve+refine cycles)
        for round_idx in range(1, self.max_rounds):
            titles = ", ".join(p["title"] for p in passages)
            reflect_resp = self.call_llm([
                {"role": "system", "content": "You evaluate answer quality. Reply with JSON only."},
                {"role": "user",   "content": REFLECT_TEMPLATE.format(
                    question=question, answer=current_answer, passages_titles=titles)},
            ], temperature=0.0)
            steps.append(f"reflect_r{round_idx+1}")

            try:
                clean = reflect_resp.strip().strip("```json").strip("```").strip()
                reflection = json.loads(clean)
            except json.JSONDecodeError:
                break  # parse failure → treat as confident, stop loop

            if reflection.get("confident", True):
                steps.append("confident_stop")
                break

            new_query = reflection.get("new_query") or question
            reason    = reflection.get("reason", "uncertain")

            passages = self.retrieve(new_query, k=self.k)
            all_passages.extend(passages)
            steps.append(f"retrieve_r{round_idx+1}")

            current_answer = self.call_llm([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": REFINE_TEMPLATE.format(
                    passages=self.format_passages(passages),
                    question=question,
                    prev_answer=current_answer,
                    reason=reason,
                )},
            ])
            steps.append(f"refine_r{round_idx+1}")

        return {
            "answer":             current_answer,
            "api_calls":          self.total_api_calls - call_count_before,
            "retrieved_passages": all_passages,
            "steps":              steps,
            "reflection_rounds":  len([s for s in steps if s.startswith("reflect")]),
        }
