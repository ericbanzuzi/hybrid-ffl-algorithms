#!/bin/bash

# Load environment variables from .env
export $(grep -v '^#' .env | xargs)

# Login to WandB
wandb login "$WANDB_KEY"

# Use SEED from environment or default
SEED_LIST="${SEED:-1}"

# Split the comma-separated list into an array
IFS=',' read -ra SEEDS <<< "$SEED_LIST"

RUN_CONFIG='\
num-server-rounds=500 \
agg-strategy="fedavg" \
cli-strategy="ditto" \
dataset="femnist" \
model="cnn" \
batch-size=32 \
learning-rate=0.1 \
local-epochs=1 \
local-iterations=1 \
lambda=1 \
fraction-fit=0.03 \
fraction-evaluate=1 \
store-client-accs=1 \
client-acc-file="ditto-femnist-accs"'

# Loop through each seed and run sequentially
for SEED_VAL in "${SEEDS[@]}"; do
  echo "🔁 Running Ditto on FEMNIST with seed=$SEED_VAL"
  flwr run . femnist-sim --run-config "$RUN_CONFIG seed=$SEED_VAL"
  echo "✅ Experiment completed for seed=$SEED_VAL!"
done
