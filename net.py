import torch
import torch.nn as nn
import torch.nn.functional as F

# input
# ball x,y,z
# prediction x,y,z
# team mate x,y,z
# opponent x,y,z
# opponent x,y,z

# output
# prediction is correct probability: j
# set to one when using training data (since prediction will be from the data frame)
# set to zero when using false data (since prediction will be from random generation)

n_dims = 3 # 3d

class Net(nn.Module):
  def __init__(self, layers):
    super(Net, self).__init__()
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

def create_model(player_count):
  """ Create a model based on player count (changes network size) """
  return Net([
    # player (x,y,z) + ball (x,y,z), n_dims = 3
    nn.Linear(player_count * n_dims + n_dims, 128), nn.ReLU(),
    nn.Linear(128,128), nn.ReLU(),
    nn.Linear(128,128), nn.ReLU(),
    nn.Linear(128,1),
  ])

def dead_test():
  print("This test only checks that the network works, not that it produces correct results.")
  for player_size in (2, 4, 6):
    net = create_model(player_size)
    net.eval()
    with torch.no_grad():
      test_data = torch.rand( player_size * n_dims + n_dims)
      inferred = net(test_data)
      print(f"{player_size} player prediction: {inferred}")

if __name__ == '__main__':
  def __main__():
    dead_test()
  __main__()
