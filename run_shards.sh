#!/bin/bash

# ============================================================
# Configuration
# ============================================================
NUM_SHARDS=50                # Total number of shards
CONCURRENCY=8                # Number of shards to run in parallel
LOG_DIR="logs"               # Directory to store per‑shard logs

# ============================================================
# Setup
# ============================================================
mkdir -p "$LOG_DIR"

# Check if the Python script exists
if [ ! -f "evaluate_dataset.py" ]; then
    echo "ERROR: evaluate_dataset.py not found in the current directory."
    exit 1
fi

echo "Starting $NUM_SHARDS shards with concurrency $CONCURRENCY..."
echo "Logs will be written to $LOG_DIR/shard_*.log"

# ============================================================
# Launch shards with xargs
# ============================================================
seq 0 $((NUM_SHARDS-1)) | xargs -P $CONCURRENCY -I {} \
    bash -c "
        python evaluate_dataset.py --shard {} --num_shards $NUM_SHARDS \
            > '$LOG_DIR/shard_{}.log' 2>&1 \
        && echo 'Shard {} finished successfully' \
        || echo 'Shard {} failed (check logs)'
    "

echo "All shards completed."