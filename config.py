from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Model
MODEL_PATH = BASE_DIR / "models" / "assistant.gguf"

# Context window
# Start conservatively with 4096. We can increase this later.
CONTEXT_SIZE = 4096

# GPU
# RTX 4060 has 8 GB VRAM.
# -1 means offload as many layers as possible.
GPU_LAYERS = -1

# Generation settings
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Number of recent messages kept in the prompt, as a hard cap.
MAX_HISTORY_MESSAGES = 20

# Rough token budget for history + system prompt + new user message.
# We estimate ~4 characters per token, which is crude but keeps us
# from silently overflowing CONTEXT_SIZE. Leaves headroom for MAX_TOKENS
# worth of response.
PROMPT_TOKEN_BUDGET = CONTEXT_SIZE - MAX_TOKENS - 64