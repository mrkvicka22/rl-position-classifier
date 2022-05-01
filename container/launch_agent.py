from collections import namedtuple
from time import sleep
import os
import uuid
import wandb
import random

if __name__ == "__main__":
  unique_name = f"container-worker-{uuid.uuid4().hex}"
  print(f"Starting {unique_name}...")
  # Set working directory to /workspace/rl-position-classifier
  os.chdir('/workspace/rl-position-classifier')
  sleep(10)

  wandb.init(entity='nevercast', project='rl-position-classifier', name=unique_name)
  api = wandb.Api()
  while True:
    # sweeps = api.project('rl-position-classifier').sweeps()
    # if len(sweeps) == 0:
    #   print("No sweeps found, waiting...")
    #   sleep(120)
    #   exit(0)
    # begin_sweep = random.choice(sweeps)
    begin_sweep = (namedtuple('Sweep', 'id')('t1r9kp3l'))
    print(f'Starting agent to service sweep/{begin_sweep.id}, will service at least 10 runs...')
    wandb.agent(begin_sweep.id, entity='nevercast', project='rl-position-classifier', count=10)
    print('Agent ended, restarting in 5 seconds...')
    sleep(5)