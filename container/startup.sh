#!/bin/bash
set -e

while true; do
  cd /workspace/rl-position-classifier
  git pull --ff-only
  WANDB_SILENT=true python3 ../launch_agent.py
  echo "Restarting in 5 seconds..."
  sleep 5
done