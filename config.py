"""Settings for the demo. Change models or prices here or with environment variables."""
import os

# Two local models served by Ollama: a small specialist and a larger generalist.
SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:1.5b")
LARGE_MODEL = os.getenv("LARGE_MODEL", "qwen2.5:7b")

# Illustrative prices in USD per 1M tokens, used only to estimate cost as if each
# tier were billed like a hosted API of its size. Replace with your provider's prices.
PRICE_PER_M = {
    "small": {"in": float(os.getenv("SMALL_PRICE_IN", 0.05)), "out": float(os.getenv("SMALL_PRICE_OUT", 0.10))},
    "large": {"in": float(os.getenv("LARGE_PRICE_IN", 2.50)), "out": float(os.getenv("LARGE_PRICE_OUT", 10.00))},
}

# Task types the router may send to the small model. Anything else goes large.
SMALL_KINDS = {"intent", "extract", "tool"}

LOG_FILE = "results/calls.jsonl"
CHART_FILE = "results/summary.png"
