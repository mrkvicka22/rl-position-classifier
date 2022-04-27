import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import load, sigmoid
import state_provider

MODEL_PATH = "model_7560000.pt"

_normal_2v2_batch = [4096, 6000] * 5 # 8192, 10240 (compensate for goal depth)

def get_state_batch(batch_size):
  batch = np.array(state_provider.get_random_play_sequence('ssl_2v2', 'test', batch_size, use_2d_map=True)) / _normal_2v2_batch
  ball, player_one, player_two, opponent_1, opponent_2 = batch[:, :2], batch[:, 2:4], batch[:, 4:6], batch[:, 6:8], batch[:, 8:10]
  return np.concatenate([ball, player_two, opponent_1, opponent_2], axis=1), player_one

def get_map(model, game_state):
  ball, player_two, opponent_1, opponent_2 = game_state[:2], game_state[2:4], game_state[4:6], game_state[6:8]
  row = np.concatenate([ball, [0, 0], player_two, opponent_1, opponent_2], axis=0)
  leftright, updown = 410, 600
  batch = np.full((410*600, 10), row)
  batch[:, 2:4] = np.array([[(x - 205) / 410, (y - 300) / 600] for x in range(leftright) for y in range(updown)])
  return sigmoid(model(torch.from_numpy(batch).float())).view(410, 600).numpy()

def dead_test():

  fig, ax = plt.subplots()

  # Create numpy array from list
  image_ref = None
  tests, label = get_state_batch(45 * 30)
  self_circle = plt.Circle((0, 0), 10, color='c')
  ball_circle = plt.Circle((0, 0), 10, color='w')
  team_player_circle = plt.Circle((0, 0), 10, color='b')
  opponent_1_circle = plt.Circle((0, 0), 10, color='y')
  opponent_2_circle = plt.Circle((0, 0), 10, color='y')

  ax.add_artist(self_circle)
  ax.add_artist(ball_circle)
  ax.add_artist(team_player_circle)
  ax.add_artist(opponent_1_circle)
  ax.add_artist(opponent_2_circle)

  model = load(MODEL_PATH)
  model.eval()

  with torch.no_grad():
    for test, label in zip(tests, label):
      test = np.array(test)
      ball_pos = test[0:2] * [205, 300] + [205, 300]
      team_player_pos = test[2:4] * [205, 300] + [205, 300]
      opponent_1_pos = test[4:6] * [205, 300] + [205, 300]
      opponent_2_pos = test[6:8] * [205, 300] + [205, 300]
      self_circle.center = label * [205, 300] + [205, 300]
      ball_circle.center = ball_pos
      team_player_circle.center = team_player_pos
      opponent_1_circle.center = opponent_1_pos
      opponent_2_circle.center = opponent_2_pos
      npimg = get_map(model, torch.from_numpy(test).float())
      # plt.imsave(f'test_{idx}.png', npimg, cmap="Greys")
      # Draw the image
      if image_ref is None:
        image_ref = ax.imshow(npimg, cmap="plasma")
      else:
        image_ref.set_data(npimg)
      fig.canvas.flush_events()
      plt.pause(1 / 30.0)

if __name__ == '__main__':
  def __main__():
    dead_test()
  __main__()