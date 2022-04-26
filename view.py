import matplotlib.pyplot as plt
import numpy as np
from torch import load
import state_provider

MODEL_PATH = "model_twos_1820000000.pt"

# Helper function for inline image display
def matplotlib_imshow(img):

  plt.pause(2)

_invert_2v2_batch = [-1, -1, 1] * 5
_normal_2v2_batch = [4096, 5120 + 900, 2044] * 5 # 8192, 10240 (compensate for goal depth)

def get_state_batch(batch_size):
  batch = np.array(state_provider.get_random_play_sequence('ssl_2v2', batch_size)) / _normal_2v2_batch
  player_one_label, team_player = batch[:, 3:5], batch[:, 6:8]
  ball = batch[:, 0:2]
  opponents_1 = batch[:, 9:11]
  opponents_2 = batch[:, 12:14]
  # input, label
  # (ball, team player, opponents), (myself)
  return np.concatenate([ball, team_player, opponents_1, opponents_2], axis=1), player_one_label[:, :2] # label only has x, y

def dead_test():
  import torch


  # Draw interactive window
  theta = np.arange(0, 2*np.pi, 0.1)
  r = 1.5

  fig, ax = plt.subplots()

  # Create numpy array from list
  image_ref = None
  tests, label = get_state_batch(45 * 30)
  self_circle = plt.Circle((0, 0), 0.1, color='c')
  ball_circle = plt.Circle((0, 0), 0.1, color='w')
  team_player_circle = plt.Circle((0, 0), 0.1, color='b')
  opponent_1_circle = plt.Circle((0, 0), 0.1, color='y')
  opponent_2_circle = plt.Circle((0, 0), 0.1, color='y')

  ax.add_artist(self_circle)
  ax.add_artist(ball_circle)
  ax.add_artist(team_player_circle)
  ax.add_artist(opponent_1_circle)
  ax.add_artist(opponent_2_circle)

  with torch.no_grad():
    for test, label in zip(tests, label):
      test = np.array(test)
      ball_pos = test[0:2] * [5, 4] + [5, 4]
      team_player_pos = test[2:4] * [5, 4] + [5, 4]
      opponent_1_pos = test[4:6] * [5, 4] + [5, 4]
      opponent_2_pos = test[6:8] * [5, 4] + [5, 4]
      self_circle.center = label * [5, 4] + [5, 4]
      ball_circle.center = ball_pos
      team_player_circle.center = team_player_pos
      opponent_1_circle.center = opponent_1_pos
      opponent_2_circle.center = opponent_2_pos
      test = torch.from_numpy(test).float()
      net = load(MODEL_PATH)
      npimg = net(test).view(8, 10)
      # plt.imsave(f'test_{idx}.png', npimg, cmap="Greys")
      # Draw the image
      if image_ref is None:
        image_ref = ax.imshow(npimg, cmap="Greys")
      else:
        image_ref.set_data(npimg)
      fig.canvas.flush_events()
      plt.pause(1.0 / 30)

if __name__ == '__main__':
  def __main__():
    dead_test()
  __main__()