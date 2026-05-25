# AgentRAG-Eval

**A systematic comparison of Naive, Self-Reflective, and Agentic RAG pipelines on HotpotQA multi-hop question answering.**

> *When does added pipeline complexity actually help — and when does it hurt?*

---

## Key Findings

Evaluated on HotpotQA (n=100, stratified by question type), using `gpt-4o-mini` + `all-MiniLM-L6-v2`:

| Pipeline | EM | F1 | API Calls | Est. Cost |
|----------|----|----|-----------|-----------|
| **Naive RAG** | **0.44** | **0.55** | **1.0×** | **$0.028** |
| Self-Reflective RAG | 0.42 | 0.54 | 3.3× | $0.064 |
| Agentic RAG | 0.37 | 0.53 | 6.2× | $0.136 |

**Counterintuitive result:** Naive RAG wins on all overall metrics. Agentic RAG uniquely recovers 8 questions Naive misses, but introduces errors on 15 questions Naive gets right (net −7). The cost-performance tradeoff is unfavorable for both complex pipelines in this setting.

**Question-type breakdown:**
- Bridge questions: Agentic ≈ Naive (0.36 vs 0.35 EM)
- Comparison questions: Naive dominates (0.72 vs Agentic 0.40 EM)

---

## Project Structure

```
agentrag-eval/
├── config.py                  # Central config (provider, model, paths)
├── .env.example               # Environment variable template
├── requirements.txt
│
├── data/
│   ├── prepare_data.py        # Download + stratified-sample HotpotQA
│   └── hotpotqa_100.json      # 100-sample dataset (generated)
│
├── index/
│   ├── build_index.py         # Embed passages → ChromaDB
│   └── chroma_db/             # Persistent vector index (generated)
│
├── pipelines/
│   ├── base.py                # Shared retrieval + LLM call (dual-provider)
│   ├── naive_rag.py           # Single retrieve → generate
│   ├── self_rag.py            # Iterative reflect → re-retrieve loop
│   └── agentic_rag.py         # Decompose → per-subQ retrieve → aggregate
│
├── eval/
│   └── metrics.py             # EM + F1 (SQuAD normalization)
│
├── experiments/
│   ├── run_experiment.py      # Main experiment runner (CLI)
│   ├── visualize.py           # Generate figures (5 plots)
│   └── failure_analysis.py    # 8-category contingency analysis
│
├── results/
│   ├── tables/                # Summary JSON per run
│   ├── raw/                   # Per-question predictions
│   ├── figures/               # PNG charts
│   └── failure_analysis.md    # Error analysis report
│
├── paper/
│   ├── references.bib         # 30 BibTeX entries
│   └── sections/              # Paper section drafts (Markdown)
│       ├── introduction.md
│       ├── related_work.md
│       ├── method.md
│       ├── experiments.md
│       └── discussion.md
│
└── daily_reports/             # Chinese research journal
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY or ANTHROPIC_API_KEY
# Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic
```

### 3. Prepare data + build index

```bash
# Download HotpotQA validation set, sample 100 questions
python data/prepare_data.py --size 100

# Embed passages and build ChromaDB index
python index/build_index.py
```

### 4. Run experiments

```bash
# Single pipeline
python experiments/run_experiment.py --pipeline naive --size 100
python experiments/run_experiment.py --pipeline self  --size 100
python experiments/run_experiment.py --pipeline agentic --size 100

# All three at once
python experiments/run_experiment.py --pipeline all --size 100
```

Results saved to `results/tables/` (summary) and `results/raw/` (per-question predictions).

### 5. Visualize results

```bash
python experiments/visualize.py
# → results/figures/fig1_overall_metrics.png
# → results/figures/fig2_stratified_em.png
# → results/figures/fig3_cost_performance.png
# → results/figures/fig4_efficiency.png
# → results/figures/fig5_f1_distribution.png
```

### 6. Failure analysis

```bash
python experiments/failure_analysis.py
# → results/failure_analysis.md
# → results/failure_analysis.json
```

---

## Pipeline Descriptions

### Naive RAG
Single retrieve → single generate. The minimal baseline: embed question → find top-5 passages → ask LLM to answer. Exactly 1 API call per question.

### Self-Reflective RAG
After the initial answer, the LLM reflects on its confidence and generates a new search query if unsatisfied. Up to 3 total rounds. Averages 3.31 API calls per question.

Inspired by: Self-RAG (Asai et al., 2024), FLARE (Jiang et al., 2023)

### Agentic RAG
Decomposes the question into sub-questions, retrieves and answers each independently, then aggregates into a final answer. Up to 5 sub-questions. Averages 6.24 API calls per question.

Inspired by: IRCoT (Trivedi et al., 2022), ReAct (Yao et al., 2023)

---

## Dual LLM Provider Support

Switch between OpenAI and Anthropic via `.env` — no code changes needed:

```bash
# Use OpenAI (default)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Use Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

---

## Reproducing the Paper Results

All experiments were run with:
- `OPENAI_MODEL=gpt-4o-mini`
- `EMBEDDING_MODEL=all-MiniLM-L6-v2`
- `TOP_K_RETRIEVE=5`
- 100-sample stratified split of HotpotQA validation set

Expected total API cost: ~$0.23 for all three pipelines × 100 questions.

---

## Citation

```bibtex
@misc{agentrag-eval-2026,
  title   = {AgentRAG-Eval: When Does Pipeline Complexity Help in Multi-hop QA?},
  author  = {Xing Yao},
  year    = {2026},
  url     = {https://github.com/xyimaging/agentrag-eval},
}
```

---

## Acknowledgements

Built with: Claude Code, [OpenAI Python SDK](https://github.com/openai/openai-python), [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python), [ChromaDB](https://www.trychroma.com/), [sentence-transformers](https://www.sbert.net/), [HotpotQA](https://hotpotqa.github.io/).
