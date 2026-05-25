"""
Agentic RAG Pipeline — decompose-retrieve-aggregate for multi-hop questions.

Breaks a complex question into sub-questions, independently retrieves
passages and generates answers for each, then synthesizes a final answer.

Inspired by:
  - ReAct (Yao et al., ICLR 2023) — interleaved reasoning and acting
  - IRCoT (Trivedi et al., ACL 2023) — retrieval interleaved with chain-of-thought
  - Agentic RAG Survey (Singh et al., 2025)
  This implementation is prompt-based (no training required).
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.base import BaseRAGPipeline
from config import TOP_K_RETRIEVE, MAX_AGENT_STEPS


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Give SHORT, DIRECT answers — typically 1-5 words."""

DECOMPOSE_TEMPLATE = """Break down the following complex question into simpler sub-questions
that can each be answered with a single Wikipedia passage.

Question: {question}

Output JSON only — a list of sub-questions (maximum {max_steps}):
{{"sub_questions": ["sub-q 1", "sub-q 2", ...]}}

Rules:
- Each sub-question should be self-contained and searchable.
- Order them logically (earlier answers may inform later ones).
- If the question is already simple (1-hop), output just one sub-question."""

SUBQ_ANSWER_TEMPLATE = """Passages:
{passages}

Sub-question: {sub_question}
(Context: this is part of answering the main question: "{main_question}")

Short answer (1-10 words):"""

AGGREGATE_TEMPLATE = """Main question: {main_question}

Sub-questions and their answers:
{sub_qa_pairs}

Based on the above sub-answers, give the final SHORT, DIRECT answer to the main question.
Typically 1-5 words:"""


class AgenticRAGPipeline(BaseRAGPipeline):
    """Decompose question → retrieve+answer each sub-question → aggregate final answer."""

    def __init__(self, collection, k: int = TOP_K_RETRIEVE,
                 max_steps: int = MAX_AGENT_STEPS):
        super().__init__(collection, name="AgenticRAG")
        self.k = k
        self.max_steps = max_steps

    def answer(self, question: str) -> dict:
        call_count_before = self.total_api_calls
        all_passages = []
        steps = []

        # Step 1: decompose question into sub-questions
        decompose_resp = self.call_llm([
            {"role": "system", "content": "You decompose complex questions. Reply with JSON only."},
            {"role": "user",   "content": DECOMPOSE_TEMPLATE.format(
                question=question, max_steps=self.max_steps)},
        ], temperature=0.0)
        steps.append("decompose")

        try:
            clean = decompose_resp.strip().strip("```json").strip("```").strip()
            sub_questions = json.loads(clean).get("sub_questions", [question])
        except (json.JSONDecodeError, AttributeError):
            sub_questions = [question]  # fall back to treating the question as atomic

        sub_questions = sub_questions[:self.max_steps]

        # Step 2: retrieve + answer each sub-question independently
        sub_qa_pairs = []
        for i, sub_q in enumerate(sub_questions):
            passages = self.retrieve(sub_q, k=self.k)
            all_passages.extend(passages)
            steps.append(f"retrieve_subq{i+1}")

            sub_answer = self.call_llm([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": SUBQ_ANSWER_TEMPLATE.format(
                    passages=self.format_passages(passages),
                    sub_question=sub_q,
                    main_question=question,
                )},
            ])
            steps.append(f"answer_subq{i+1}")
            sub_qa_pairs.append(f"Q{i+1}: {sub_q}\nA{i+1}: {sub_answer}")

        # Step 3: aggregate sub-answers into final answer
        sub_qa_text = "\n\n".join(sub_qa_pairs)
        final_answer = self.call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": AGGREGATE_TEMPLATE.format(
                main_question=question, sub_qa_pairs=sub_qa_text)},
        ])
        steps.append("aggregate")

        return {
            "answer":             final_answer,
            "api_calls":          self.total_api_calls - call_count_before,
            "retrieved_passages": all_passages,
            "steps":              steps,
            "sub_questions":      sub_questions,
            "sub_qa_pairs":       sub_qa_pairs,
            "n_sub_questions":    len(sub_questions),
        }
