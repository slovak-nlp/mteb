#!/usr/bin/env python
"""
Minimal script to run MTEB(slk, v1) benchmark against custom OpenAI endpoint.

Setup:
    pip install mteb[openai]
    
    # Set environment variables
    export OPENAI_BASE_URL="https://your-endpoint.com/v1"
    export OPENAI_API_KEY="your-api-key"

Usage:
    # Small model (1536 dims)
    python run_slovak_benchmark.py

    # Large model (3072 dims)
    python run_slovak_benchmark.py --model text-embedding-3-large --embed-dim 3072

    # Custom output directory
    python run_slovak_benchmark.py --output-dir my_results
"""
import argparse
import os
import mteb
from openai import OpenAI
from mteb.models.openai_models import OpenAIWrapper
from mteb.model_meta import ModelMeta

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Run MTEB Slovak benchmark on OpenAI embeddings")
parser.add_argument("--model", default="text-embedding-3-small", 
                    help="Model name (default: text-embedding-3-small)")
parser.add_argument("--embed-dim", type=int, default=1536,
                    help="Embedding dimension (default: 1536)")
parser.add_argument("--output-dir", default="results",
                    help="Output directory (default: results)")
args = parser.parse_args()

# Configuration from environment variables
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

if not BASE_URL:
    print("Error: OPENAI_BASE_URL environment variable is not set")
    exit(1)
if not API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set")
    exit(1)

MODEL = args.model
EMBED_DIM = args.embed_dim
OUTPUT_DIR = args.output_dir

print(f"Configuration:")
print(f"  Base URL: {BASE_URL}")
print(f"  Model: {MODEL}")
print(f"  Embedding Dim: {EMBED_DIM}")
print(f"  Output Dir: {OUTPUT_DIR}")
print()

# Create client and model
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
model = OpenAIWrapper(
    model_name=MODEL,
    max_tokens=8191,
    embed_dim=EMBED_DIM,
    client=client,
)

# Set model metadata for proper directory naming
model.mteb_model_meta = ModelMeta(
    name=f"openai/{MODEL}",
    revision=f"dim{EMBED_DIM}",
    release_date=None,
    languages=None,
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=8191,
    embed_dim=EMBED_DIM,
    license=None,
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    framework=["API"],
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
)

# Load Slovak benchmark and run
benchmark = mteb.get_benchmark("MTEB(slk, v1)")
evaluation = mteb.MTEB(tasks=benchmark)
results = evaluation.run(model, output_folder=OUTPUT_DIR, verbosity=2)

print(f"\n✓ Results saved to: {OUTPUT_DIR}")
