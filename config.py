import os
from dotenv import load_dotenv

load_dotenv()

# Provider selection: "openai" or "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# OpenAI
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

# Anthropic
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL      = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_JUDGE_MODEL = os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-haiku-4-5-20251001")

EXPERIMENT_SAMPLE_SIZE = int(os.getenv("EXPERIMENT_SAMPLE_SIZE", "100"))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
INDEX_DIR    = os.path.join(PROJECT_ROOT, "index")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR     = os.path.join(PROJECT_ROOT, "logs")

CHROMA_COLLECTION_NAME = "hotpotqa_contexts"
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"

TOP_K_RETRIEVE       = 5
MAX_REFLECTION_ROUNDS = 3
MAX_AGENT_STEPS       = 5
