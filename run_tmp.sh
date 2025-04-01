#!/bin/bash
# Auxiliary script to run the training process with different configurations sequentially.
configs=("configs/td_10ms.toml" "configs/td_10ms_shortfilt.toml")
for config in "${configs[@]}"; do
    python train.py -d 2 -c "$config"
done