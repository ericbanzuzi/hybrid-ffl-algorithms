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
agg-strategy="fedprox" \
cli-strategy="fedprox" \
dataset="femnist" \
model="cnn" \
batch-size=20 \
learning-rate=0.1 \
fraction-fit=0.03 \
proximal-mu=0.01 \
fraction-evaluate=1 \
store-client-accs=1 \
client-acc-file="femnist/fedprox-femnist-accs"'

# Loop through each seed and run sequentially
for SEED_VAL in "${SEEDS[@]}"; do
  echo "🔁 Running FedProx on FEMNIST with seed=$SEED_VAL"
  flwr run . femnist-sim --run-config "$RUN_CONFIG seed=$SEED_VAL"
  echo "✅ Experiment completed for seed=$SEED_VAL!"
done
