import os
from random import random
import torch
from torch import save, load
from torch.optim import SGD
from torch.nn import MSELoss
from net import Net, LAYERS_TWOS
from settings import TWOS_MODEL_PATH
import numpy as np

from state_provider import get_replay_batch


MODEL_PATH = TWOS_MODEL_PATH

def train_step(net, optimiser, loss_fn, inputs, targets):
  # optimiser.zero_grad()
  for param in net.parameters():
    param.grad = None
  y_pred = net(inputs)
  loss = loss_fn(y_pred, targets)
  loss.backward()
  optimiser.step()
  return loss.item()

_invert_2v2_batch = [-1, -1, 1] * 5
_normal_2v2_batch = [4096, 5120 + 900, 2044] * 5 # 8192, 10240 (compensate for goal depth)

def world_pos_to_map_pos(train_labels):
  return (train_labels * [4, 5] + [4, 5]).astype(np.int32)

def labels_to_map(labels):
  for x, y in world_pos_to_map_pos(labels):
    label = np.full((8, 10), 0.5, dtype=np.float32)

    # clamp to map size
    x = max(0, min(7, x))
    y = max(0, min(9, y))
    label[x, y] = 0
    yield label

def get_state_batch(batch_size):
  batch = np.array(get_replay_batch('ssl_2v2', batch_size)) / _normal_2v2_batch
  if random() < 0.5: # flip teams
    batch *= _invert_2v2_batch
  if random() < 0.5: # swap first and second player
    player_one_label, team_player = batch[:, 6:9], batch[:, 3:6]
  else:
    player_one_label, team_player = batch[:, 3:6], batch[:, 6:9]
  ball = batch[:, 0:3]
  opponent_1 = batch[:, 9:12]
  opponent_2 = batch[:, 12:15]
  # input, label
  # (ball, team player, opponents), (myself)
  return np.concatenate([ball[:, :2], team_player[:, :2], opponent_1[:, :2], opponent_2[:, :2]], axis=1), player_one_label[:, :2] # label only has x, y

def train():
  model = Net(LAYERS_TWOS)
  steps_taken = 0

  model_path_base = MODEL_PATH.replace('.pt', '')
  load_steps = 23_000_000
  model_load_path = f'{model_path_base}_{load_steps}.pt'
  if os.path.exists(model_load_path):
    model = load(model_load_path)
    steps_taken = load_steps

  _optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  # Decrease learning rate by 0.1 every 10 epochs
  # scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=10, gamma=0.1)

  # Create mean squared error loss function
  # loss_fn = MSELoss()
  # Use cross entropy loss function
  loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

  # Batch size increases after the first 1 million steps
  batch_size = 1_000_000

  # Train
  while True:

    train_features, train_label_pos = get_state_batch(batch_size)
    train_labels = np.array(list(labels_to_map(train_label_pos.astype(np.float32))))
    labels = torch.tensor(train_labels).view((batch_size, 80))
    inputs = torch.tensor(train_features.astype(np.float32))
    train_step(model, _optimizer, loss_fn, inputs, labels)

    steps_taken += batch_size
    # print a message every 1000 steps
    if steps_taken % 1_000 == 0:
      print(f"Steps completed: {steps_taken}")

    # Validate and save checkpoint every 1 million steps
    if steps_taken % 10_000_000 == 0:
      model.eval()
      with torch.no_grad():
        # Validate
        val_features, val_label_pos = get_state_batch(100_000)
        val_labels = np.array(list(labels_to_map(val_label_pos.astype(np.float32))))
        val_inputs = torch.tensor(val_features.astype(np.float32))
        val_labels = torch.tensor(val_labels).view((len(val_features), 80))
        val_loss = loss_fn(model(val_inputs), val_labels)
        print(f'Validation loss: {val_loss}')
        # Step scheduler
        # scheduler.step()
        print(f'Learning rate: {_optimizer.param_groups[0]["lr"]}')
        # Save model
        model_path = f'{model_path_base}_{steps_taken}.pt'
        save(model, model_path)
        print(f'Model saved to {model_path}')
        model.train()

if __name__ == '__main__':
  def __main__():
    train()
  __main__()