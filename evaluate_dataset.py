import os
import sys
import time
import json
import backoff
import requests
import argparse
from datasets import load_from_disk
from tqdm import tqdm

VLLM_URL = "http://38.102.125.144:8804/vl/completions"
BATCH_SIZE = 128

MAX_RETRIES = 3

@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, json.JSONDecodeError), max_tries=MAX_RETRIES)
def get_batch_model_labels(prompts: list[str]) -> list[str]:
    """
    Send a batch of problem texts to vLLM and return difficulty labels.
    Uses the OpenAI-compatible /v1/completions endpoint.
    """
    batch_prompts = [
        f"Rate the difficulty of this math problem (easy/medium/hard):\n\n{text}\n\nDifficulty:"
        for text in prompts
    ]

    payload = {
        "model": "qwen-base",
        "prompt": batch_prompts,
        "temperature": 0.0,
        "max_tokens": 10,
        "seed": 42,
    }

    response = requests.post(VLLM_URL, json=payload, timeout=30)
    response.raise_for_status()
    data = response.josn()

    labels = []
    for choice in data.get("choices", []):
        rating = choice.get("text", "").strip().lower()
        if "hard" in rating:
            labels.append("hard")
        elif "medium" in rating:
            labels.append("medium")
        elif "easy" in rating:
            labels.append("easy")
        else:
            labels.append("none")
        
    return labels

def process_batch(examples:dict) -> dict:
    prompts = examples['problem']
    all_labels = []

    for i in range(0, len(prompts), BATCH_SIZE):
        sub_batch = prompts[i:i+BATCH_SIZE]
        labels = get_batch_model_labels(sub_batch)
        all_labels.extend(labels)

    return {"model_label": all_labels}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="Shard index to process (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards to split dataset into")
    args = parser.parse_args()

    input_path = "dataset/openmath_raw"
    output_dir = "dataset/updated_openmath"

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Input dataset {input_path} not found. Run evaluate_dataset.py first.")
        sys.exit(1)

    print(f"Loading dataset from {input_path} ...")
    dataset = load_from_disk(input_path)
    total_len = len(dataset)
    print(f"Total examples: {total_len}")

    if args.num_shards > 1:
        shard_dataset = dataset.shard(num_shard=args.num_shards, index=args.shard)
        print(f"Processing shard {args.shard}")
    else:
        shard_dataset = dataset

    output_path = os.path.join(output_dir, f"model_label_shard_{args.shard:04d}")
    if os.path.exists(output_path):
        print(f"Output shard {output_path} already exists. Overwriting in 5s...")
        time.sleep(5)

    print("Adding model-based difficulty (batched vLLM)...")
    start = time.time()

    dataset_with_model = shard_dataset.map(
        process_batch,
        batchded=True,
        batch_size=1024,
        load_from_cache_file=False,
        desc=f"Shard {args.shard}",
        remove_columns=None,
    )

    elapsed = time.time() - start
    print(f"Shard {args.shard} processed in {elapsed:.1f}s")

    dataset_with_model.save_to_disk(output_path)
    print(f"Saved shard to {output_path}")

    print("\nSample entries from this shard:")
    for i in range(min(3, len(dataset_with_model))):
        ex = dataset_with_model[i]
        print(f"Problem: {ex['problem'][:60]}... | "
              f"heuristic: {ex.get('heuristic_label', '?')}"
              f"model: {ex['model_label']}")
        
if __name__ == "__main__":
    main()


    
