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
agg-strategy="qfedavg" \
cli-strategy="fedavg" \
dataset="shakespeare" \
model="lstm" \
batch-size=10 \
learning-rate=0.8 \
agg-learning-rate=0.8 \
fraction-fit=0.33 \
fraction-evaluate=1 \
store-client-accs=1 \
qparam=0.001 \
client-acc-file="shakespeare/qfedavg-text-accs-q001"'

# Loop through each seed and run sequentially
for SEED_VAL in "${SEEDS[@]}"; do
  echo "🔁 Running q-FedAvg on SHAKESPEARE with seed=$SEED_VAL"
  flwr run . text-sim --run-config "$RUN_CONFIG seed=$SEED_VAL"
  echo "✅ Experiment completed for seed=$SEED_VAL!"
done