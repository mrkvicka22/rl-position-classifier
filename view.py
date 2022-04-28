import matplotlib
import matplotlib.cm as cmx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import load, sigmoid
import train
import state_provider
from mpl_toolkits.mplot3d.art3d import juggle_axes

MODEL_PATH = "model_ssl_2v2_latest.pt"


def get_state_batch(batch_size):
  normal = train.normalization(4, use_2d_map=False)
  return np.array(state_provider.get_random_play_sequence('ssl_2v2', 'test', batch_size, use_2d_map=False)) / normal

def generate_volume_points(width_steps, depth_steps, height_steps):
  return np.linspace(-1, 1, width_steps), np.linspace(-1, 1, depth_steps), np.linspace(0, 1, height_steps)

def generate_volume_batch(width_steps, depth_steps, height_steps):
  xs, ys, zs = generate_volume_points(width_steps, depth_steps, height_steps)
  return np.array([[x, y, z] for x in xs for y in ys for z in zs])

def infer_point_cloud_from_volume(volume, volume_space_shape, game_state, model):
  # point replaces game_state[3:6] for every point in volume
  # volume is a numpy array of points
  x = np.array([np.concatenate([game_state[0:3], point, game_state[6:]]) for point in volume])
  # Pass x through torch model (10, 10, 10, 15) -> (10, 10, 10, 1)
  y = model(torch.from_numpy(x).float())
  return sigmoid(y).reshape(volume_space_shape + (1,)).numpy()


def dead_test():

  # cm = plt.get_cmap()
  cm = LinearSegmentedColormap.from_list('', ['white', 'white', 'red'])
  cNorm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
  scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  states = get_state_batch(45 * 30)[::3, :]
  model = load(MODEL_PATH)
  model.eval()

  size = 20
  points = generate_volume_batch(size, size, size)

  scatter_ref = None
  entity_ref = None

  with torch.no_grad():
    while True:
      for test_state in states:
        # Get point cloud from volume
        point_cloud = infer_point_cloud_from_volume(points, (size, size, size), test_state, model)

        entity_positions = [
          test_state[0::3],
          test_state[1::3],
          test_state[2::3]
        ]

        # Draw point cloud
        if entity_ref is None:
          entity_ref = ax.scatter(entity_positions[0], entity_positions[1], entity_positions[2], c=['black', 'cyan', 'blue', 'orange', 'orange'], s=20)
          scatter_ref = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_cloud[:, :, :, 0].reshape((-1)), cmap=cm, s=1)
          fig.colorbar(scalarMap)
        else:
          scatter_ref.set_array(point_cloud[:, :, :, 0].reshape((-1)))
          entity_ref._offsets3d = tuple(entity_positions)
          fig.canvas.draw()
          plt.pause(1 / 30)

if __name__ == '__main__':
  def __main__():
    dead_test()
  __main__()