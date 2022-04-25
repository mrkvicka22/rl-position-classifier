import matplotlib.pyplot as plt
import numpy as np
from torch import load
import state_provider

MODEL_PATH = "model_twos_34000000.pt"

# Helper function for inline image display
def matplotlib_imshow(img):

  plt.pause(2)

_invert_2v2_batch = [-1, -1, 1] * 5
_normal_2v2_batch = [4096, 5120 + 900, 2044] * 5 # 8192, 10240 (compensate for goal depth)

def get_state_batch(batch_size):
  batch = np.array(state_provider.get_replay_batch('ssl_2v2', batch_size)) / _normal_2v2_batch
  player_one_label, team_player = batch[:, 3:6], batch[:, 6:9]
  ball = batch[:, 0:3]
  opponents = batch[:, 9:]
  # input, label
  # (ball, team player, opponents), (myself)
  return np.concatenate([ball, team_player, opponents], axis=1), player_one_label[:, :2] # label only has x, y

def dead_test():
  import torch


  # Draw interactive window
  theta = np.arange(0, 2*np.pi, 0.1)
  r = 1.5

  fig, ax = plt.subplots()

  # Create numpy array from list
  image_ref = None
  tests, label = get_state_batch(100)
  with torch.no_grad():
    for idx, test in enumerate(tests):
      test = np.array(test)
      test = torch.from_numpy(test).float()
      print(test)
      net = load(MODEL_PATH)
      npimg = net(test).view(80, 100)
      # plt.imsave(f'test_{idx}.png', npimg, cmap="Greys")
      # Draw the image
      if image_ref is None:
        image_ref = ax.imshow(npimg, cmap="Greys")
      else:
        image_ref.set_data(npimg)
      fig.canvas.flush_events()
      plt.pause(0.01)

if __name__ == '__main__':
  def __main__():
    dead_test()
  __main__()