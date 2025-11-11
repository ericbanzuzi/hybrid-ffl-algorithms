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
num-server-rounds=100 \
agg-strategy="fedprox" \
cli-strategy="fedprox" \
dataset="cifar10" \
model="cnn-cifar" \
batch-size=64 \
learning-rate=0.01 \
proximal-mu=0.001 \
fraction-fit=1 \
fraction-evaluate=1 \
store-client-accs=1 \
client-acc-file="cifar10/fedprox-cifar10-accs-cnn-cifar"'

# Loop through each seed and run sequentially
for SEED_VAL in "${SEEDS[@]}"; do
  echo "🔁 Running FedProx on CIFAR10 with seed=$SEED_VAL"
  flwr run . cifar10-sim --run-config "$RUN_CONFIG seed=$SEED_VAL"
  echo "✅ Experiment completed for seed=$SEED_VAL!"
done
