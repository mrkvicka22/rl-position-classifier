import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from state_provider import get_random_play_sequence

space = (256, 375)

def create_gamestates(space, player_count, ball_state=None, player_positions=None):
  """ Create a batch of game states with all zeros except for the prediction position.
  """
  ndims = len(space)
  mgrid_slices = [slice(-1, 1, 1j * steps) for steps in space]
  predictions = np.mgrid[mgrid_slices].reshape(ndims, -1).T
  empty_entity = np.zeros((np.prod(space), ndims))
  ball_state = np.full((predictions.shape[0], ndims), ball_state) if ball_state is not None else empty_entity
  player_positions = np.repeat(empty_entity, player_count - 1, axis=1) if player_positions is None else \
                     np.full((predictions.shape[0], len(player_positions)), player_positions)
  game_states = np.concatenate([ball_state, predictions, player_positions], axis=1)
  return game_states

def mock_model(space, game_states):
  """ Create a batch of game states with all zeros except for the prediction position.
  """
  ndims = len(space)
  predictions = game_states[:, ndims:2 * ndims]
  # Calculate the distance of each prediction from the center of the map, scale by space to compensate for aspect ratio
  output = 1 - (np.linalg.norm(predictions * np.array(space), axis = 1) / (np.sqrt(ndims) * np.max(space)))
  return output

def apply_model(model, game_states):
  with torch.no_grad():
    output = torch.sigmoid(model(torch.tensor(game_states).float())).numpy()
  return output

def transform_output_into_space(output, space):
  """ Transform the output into a space of the same size.
  """
  return np.reshape(output, space)

def display_graphical_heatmap(graphical, cmap='viridis', focus=2):
  """ Display a heatmap of the output.
  """
  plt.imshow(graphical ** focus, cmap=cmap, interpolation='nearest')
  plt.show()


def render_graphical_heatmap(graphical, arena: Image, mask: Image, cmap='viridis', focus=2, normalize=True, heatmap_opacity = 0.5, entities = None, image_size=1):
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
  if entities:
    draw = ImageDraw.Draw(heatmap_image)
    for position, color in entities:
      position = tuple(position.astype(int))
      # swap position coords
      position = position[1], position[0]
      entity_size = (2, 2)
      box = (position[0] - entity_size[0], position[1] - entity_size[1], position[0] + entity_size[0], position[1] + entity_size[1])
      if len(color) == 2:
        fill, outline = color
      else:
        fill, outline = color, color
      draw.ellipse(box, fill=fill, outline=outline)
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
  if image_size != 1:
    combined = combined.resize((int(combined.width * image_size), int(combined.height * image_size)), resample=Image.Resampling.LANCZOS)
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

def world_space_to_image_space(position, space):
  """ Convert a position in world space to image space.
  """
  # This doesn't account for the way that space is scaled with aspect ratios
  # This could be fixed by scaling the space to the arena size, TODO later.
  # Clamp to space
  return np.clip(position * np.array(space) // 2 + np.array(space) // 2, 0, space)


def create_image_stream(model, state_generator, space, arena, mask, image_size=1):
  for game_state, frame in state_generator:
    outputs = apply_model(model, game_state)
    graphical = transform_output_into_space(outputs, space)

    entities = [
      # ball
      (world_space_to_image_space(frame[:2], space), ((50, 50, 50, 50), (255, 0, 255, 255))),
      # first player
      (world_space_to_image_space(frame[2:4], space), (0, 255, 255, 255)),
      # second player
      (world_space_to_image_space(frame[4:6], space), (0, 0, 255, 255)),
      # third player
      (world_space_to_image_space(frame[6:8], space), (255, 168, 0, 255)),
      # fourth player
      (world_space_to_image_space(frame[8:10], space), (255, 168, 0, 255)),
    ]
    yield render_graphical_heatmap(graphical, arena, mask, focus=3, cmap='magma', heatmap_opacity=0.6, image_size=image_size, entities=entities)

def create_animation_from_model(model_path, image_path, player_count, image_size=1):
  arena, mask = load_arena()
  model = torch.load(model_path)
  model.eval()

  fps = 4 # get 2 seconds at 4 fps
  seconds = 5
  batch_frames = seconds * 15
  sequence = get_random_play_sequence('ssl_2v2', 'test', batch_size=batch_frames, use_2d_map=True)[::15 // fps]
  sequence = np.array(sequence) / ([4096, 6000] * (1 + player_count))

  game_state_gen = ((create_gamestates(space, player_count=4, ball_state=frame[:2], player_positions=frame[4:]), frame) for frame in sequence)

  images = list(create_image_stream(model, game_state_gen, space, arena, mask, image_size=image_size))
  # Save all images as a gif
  images[0].save(image_path, save_all=True, append_images=images[1:], duration=seconds, loop=0)

if __name__ == '__main__':
  arena, mask = load_arena()
  model = torch.load('renderer.pt')
  model.eval()
  player_count = 4

  fps = 4 # get 2 seconds at 4 fps
  seconds = 5
  batch_frames = seconds * 15

  for _ in range(10):
    sequence = get_random_play_sequence('ssl_2v2', 'test', batch_size=batch_frames, use_2d_map=True)[::15 // fps]
    sequence = np.array(sequence) / ([4096, 6000] * (1 + player_count))

    game_state_gen = ((create_gamestates(space, player_count=4, ball_state=frame[:2], player_positions=frame[4:]), frame) for frame in sequence)

    images = list(create_image_stream(model, game_state_gen, space, arena, mask))
    # Save all images as a gif
    images[0].save(f'renderer_{_}.gif', save_all=True, append_images=images[1:], duration=seconds, loop=0)


  # outputs = mock_model(space, game_states)
  # graphical = transform_output_into_space(outputs, space)
  # image = render_graphical_heatmap(graphical, arena, mask, focus=3, cmap='magma', heatmap_opacity=0.6)
  # save_heatmap_image('heatmap_mock.png', image)


