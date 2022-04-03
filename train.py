import torch
from torch import save, load
from torch.optim import SGD
from torch.nn import MSELoss
from net import Net, LAYERS_TWOS
from settings import TWOS_MODEL_PATH
import numpy as np

MODEL_PATH = TWOS_MODEL_PATH

def train_step(net, optimiser, loss_fn, inputs, targets):
  optimiser.zero_grad()
  y_pred = net(inputs)
  loss = loss_fn(y_pred, targets)
  # Print error
  print(loss)
  loss.backward()
  optimiser.step()
  return loss.item()

def train():
  model = Net(LAYERS_TWOS)
  _optimizer = SGD(model.parameters(), lr=1, momentum=0.9)
  # Create mean squared error loss function
  loss_fn = MSELoss()

  # Create 80x100 numpy array with one hot value to test model
  target_data = torch.ones(80, 100)
  target_data[40, 40] = 0
  target_data = target_data.view(8000)

  # Create batches
  batch_size = 1000
  inputs = torch.rand((batch_size, 12))
  targets = np.full((batch_size, 8000), target_data).astype(np.float32)

  # Create tensor from tuple
  inputs = torch.tensor(inputs)
  targets = torch.tensor(targets)

  # Train
  for i in range(0, 100_000, batch_size):
    train_step(model, _optimizer, loss_fn, inputs, targets)
    # print a message every 1000 steps
    if i % 1000 == 0:
      print(f"Step {i}")

  save(model, MODEL_PATH)
  model = load(MODEL_PATH)

if __name__ == '__main__':
  def __main__():
    train()
  __main__()