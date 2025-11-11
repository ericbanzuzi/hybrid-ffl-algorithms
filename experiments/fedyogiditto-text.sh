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
agg-strategy="fedyogi" \
cli-strategy="ditto" \
dataset="shakespeare" \
model="lstm" \
batch-size=10 \
learning-rate=0.8 \
agg-learning-rate=0.01 \
lambda=0.1 \
fraction-fit=0.33 \
fraction-evaluate=1 \
store-client-accs=1 \
client-acc-file="shakespeare/fedyogiditto-text-accs"'

# Loop through each seed and run sequentially
for SEED_VAL in "${SEEDS[@]}"; do
  echo "🔁 Running FedYogiDitto on SHAKESPEARE with seed=$SEED_VAL"
  flwr run . text-sim --run-config "$RUN_CONFIG seed=$SEED_VAL"
  echo "✅ Experiment completed for seed=$SEED_VAL!"
done