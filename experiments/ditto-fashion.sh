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
dataset="fashion-mnist" \
model="cnn" \
batch-size=20 \
learning-rate=0.05 \
local-epochs=1 \
local-iterations=1 \
lambda=1 \
fraction-fit=0.05 \
fraction-evaluate=1 \
store-client-accs=1 \
client-acc-file="fashion-mnist/ditto-fashion-accs"'

# Loop through each seed and run sequentially
for SEED_VAL in "${SEEDS[@]}"; do
  echo "🔁 Running Ditto on Fashion-MNIST with seed=$SEED_VAL"
  flwr run . fashion-mnist-sim --run-config "$RUN_CONFIG seed=$SEED_VAL"
  echo "✅ Experiment completed for seed=$SEED_VAL!"
done
