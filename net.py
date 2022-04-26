import torch
import torch.nn as nn
import torch.nn.functional as F

# input
# ball x,y
# team mate x,y
# opponent x,y
# opponent x,y

# output
# heatmap
# game maybe is 10k by 8k in size, so will use
# 100 x 80 elements to indicate position

LAYERS_ONES = [
  nn.Linear(8,128), nn.ReLU(),
  nn.Linear(128,128), nn.ReLU(),
  nn.Linear(128,128), nn.ReLU(),
  nn.Linear(128,80)
]

LAYERS_TWOS = [
  nn.Linear(8,128), nn.ReLU(),
  nn.Linear(128,128), nn.ReLU(),
  nn.Linear(128,128), nn.ReLU(),
  nn.Linear(128,80)
]

LAYERS_THREES = [
  nn.Linear(18,128), nn.ReLU(),
  nn.Linear(128,128), nn.ReLU(),
  nn.Linear(128,128), nn.ReLU(),
  nn.Linear(128,8000)
]

class Net(nn.Module):
  def __init__(self, layers):
    super(Net, self).__init__()
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

def dead_test():
  test_data = torch.rand(12)
  net = Net(LAYERS_TWOS)
  inferred = net(test_data)
  print(inferred.size())
  print(inferred)

if __name__ == '__main__':
  def __main__():
    dead_test()
  __main__()
