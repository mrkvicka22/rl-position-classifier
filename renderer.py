import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_gamestates_and_output(space, player_count):
  """ Create a batch of game states with all zeros except for the prediction position.
  """
  ndims = len(space)
  mgrid_slices = [slice(-1, 1, 1j * steps) for steps in space]
  predictions = np.mgrid[mgrid_slices].reshape(ndims, -1).T
  empty_entity = np.zeros((np.prod(space), ndims))
  game_states = np.concatenate([empty_entity, predictions, np.repeat(empty_entity, player_count - 1, axis=1)], axis=1)
  # Calculate the distance of each prediction from the center of the map, scale by space to compensate for aspect ratio
  output = 1 - (np.linalg.norm(predictions * np.array(space), axis = 1) / (np.sqrt(ndims) * np.max(space)))
  return game_states, output

def transform_output_into_space(output, space):
  """ Transform the output into a space of the same size.
  """
  return np.reshape(output, space)

def display_graphical_heatmap(graphical, cmap='viridis', focus=2):
  """ Display a heatmap of the output.
  """
  plt.imshow(graphical ** focus, cmap=cmap, interpolation='nearest')
  plt.show()


def render_graphical_heatmap(graphical, arena: Image, mask: Image, cmap='viridis', focus=2, normalize=True, heatmap_opacity = 0.5):
  """ Render a heatmap of the output to a numpy array for storage or sending to wandb.
  """
  cmap = plt.cm.get_cmap(cmap)
  graphical **= focus
  if normalize:
    graphical = plt.Normalize(vmin=graphical.min(), vmax=graphical.max())(graphical)
  # 2d array of RGBA values
  heatmap_graphical = cmap(graphical)
  # Create a PIL image from the heatmap array of RGBA pixels
  heatmap_image = Image.frombuffer(mode='RGBA', size=(graphical.shape[1], graphical.shape[0]), data=(heatmap_graphical * 255).astype(np.uint8))
  # Calculate new image width by scaling to arena.height, maintaining aspect ratio
  new_width = int(arena.height * graphical.shape[1] / graphical.shape[0])
  heatmap_image = heatmap_image.resize((new_width, arena.height), resample=Image.Resampling.LANCZOS)
  # If the heatmap and arena aren't the same aspect ratio,
  # we need to create a new canvas the size of the arena to paste the heatmap on
  if new_width != arena.width:
    # Make the heatmap_image canvas size the same as arena
    temp_image = Image.new('RGBA', (arena.width, arena.height), (0, 0, 0, 0))
    # Paste heatmap_image into temp_image, relative to the image center
    temp_image.paste(heatmap_image, (arena.width // 2 - new_width // 2, arena.height // 2 - arena.height // 2), heatmap_image)
    heatmap_image = temp_image
  # Set the heatmap alpha channel to the mask
  heatmap_image.putalpha(mask)
  # Blend the heatmap with the arena
  combined = Image.blend(arena, heatmap_image, heatmap_opacity)
  return combined

def save_heatmap_image(filename, image):
  """ Save a heatmap image to a file.
  """
  image.save(filename)

def load_arena():
  """ Load the arena image, and arena mask image.
  """
  arena = Image.open('arena.png')
  arena_mask = Image.open('arena_mask.png')
  # Convert the mask to L1 alpha mask
  arena_mask = arena_mask.convert('L')
  return arena, arena_mask


if __name__ == '__main__':
  space = (400, 600)
  arena, mask = load_arena()
  states, outputs = create_gamestates_and_output(space, player_count=4)
  graphical = transform_output_into_space(outputs, space)
  image = render_graphical_heatmap(graphical, arena, mask, focus=3, cmap='magma', heatmap_opacity=0.6)
  save_heatmap_image('heatmap.png', image)
