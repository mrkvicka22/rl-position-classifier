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

class Net(nn.Module):
  def __init__(self, layers):
    super(Net, self).__init__()
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

def create_model(player_count, ndims = 3, hidden_layers = 2, hidden_units = 128):
  """ Create a model based on player count (changes network size) """
  # player (x,y,z) + ball (x,y,z), ndims = 3 when 3d (x, y, z), else 2 for (x, y)
  return Net(
    [nn.Linear(player_count * ndims + ndims, hidden_units), nn.ReLU()] +
    [nn.Linear(hidden_units, hidden_units), nn.ReLU()] * hidden_layers +
    [nn.Linear(hidden_units,1)]
  )

def dead_test():
  print("This test only checks that the network works, not that it produces correct results.")
  for ndims in [2, 3]:
    for player_size in (2, 4, 6):
      net = create_model(player_size, ndims)
      net.eval()
      with torch.no_grad():
        test_data = torch.rand( player_size * ndims + ndims)
        inferred = net(test_data)
        print(f"{ndims}d {player_size} player prediction: {inferred}")

if __name__ == '__main__':
  def __main__():
    dead_test()
  __main__()
