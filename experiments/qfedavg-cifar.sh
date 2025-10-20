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
num-server-rounds=2 \
agg-strategy="qfedavg" \
cli-strategy="fedavg" \
dataset="cifar10" \
model="resnet18" \
batch-size=32 \
learning-rate=0.01 \
agg-learning-rate=0.01 \
fraction-fit=1 \
fraction-evaluate=1 \
group-norm=1 \
store-client-accs=1 \
qparam = 10 \
client-acc-file="qfedavg-cifar10-accs-res-gn"'

# Loop through each seed and run sequentially
for SEED_VAL in "${SEEDS[@]}"; do
  echo "🔁 Running q-FedAvg on CIFAR10 with seed=$SEED_VAL"
  flwr run . cifar10-sim --run-config "$RUN_CONFIG seed=$SEED_VAL"
  echo "✅ Experiment completed for seed=$SEED_VAL!"
done
