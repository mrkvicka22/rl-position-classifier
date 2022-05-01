#!/bin/bash
set -e

apt-get install -y python3-pip git python-is-python3
cd /workspace

cp /workspace/.netrc ~/.netrc

# If rl-position-classifier is not installed, install it
if [ ! -d "/workspace/rl-position-classifier" ]; then
    git clone https://github.com/nevercast/rl-position-classifier.git --depth 1
fi

python -m pip install -r /workspace/rl-position-classifier/requirements.txt

while true; do
  cd /workspace/rl-position-classifier
  git pull --ff-only
  WANDB_SILENT=true python ../launch_agent.py
  echo "Restarting in 5 seconds..."
  sleep 5
done
