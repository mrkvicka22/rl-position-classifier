import sqlite3

import torch
import numpy as np
from PIL import Image, ImageDraw
from nets import DiscriminatorNet

def load_arena():
  """ Load the arena image, and arena mask image.
  """
  arena = Image.open('arena.png')
  arena_mask = Image.open('arena_mask.png')
  # Convert the mask to L1 alpha mask
  arena_mask = arena_mask.convert('L')
  return arena, arena_mask

def get_columns(con):
    cursor = con.cursor()
    helper_data = cursor.execute(f"SELECT * FROM clean_ssl_2v2_validation")
    cursor.close()
    column_names = list(map(lambda x: x[0], helper_data.description))  # get the names of columns
    return [name for name in column_names if (name != "id" and name != "file_id")]

def get_random_play_sequence(gamemode, suffix, batch_size, use_2d_map=False, verbose=False):
  con = sqlite3.connect('replays-waddles.db')
  cur = con.cursor()
  columns = get_columns(con)
  expression =f'''
  select {",".join(columns)} from clean_ssl_2v2_validation limit {batch_size} offset (
      select (1+abs(random()) % ((SELECT rowid FROM clean_ssl_2v2_validation ORDER BY rowid DESC LIMIT 1) - {batch_size})) from clean_ssl_2v2_validation limit 1
  );
  '''
  if verbose:
    print(expression)
  cur.execute(expression)
  return cur.fetchall()

if __name__ == '__main__':
    space = (256, 375)
    arena, mask = load_arena()
    model = DiscriminatorNet()
    model.load_state_dict(torch.load('GAN_models_3.pt')['Discriminator_state_dict'])
    model.eval()
    player_count = 4

    fps = 4  # get 2 seconds at 4 fps
    seconds = 15
    batch_frames = seconds * 15

    for _ in range(10):
        sequence = get_random_play_sequence('ssl_2v2', 'test', batch_size=batch_frames, use_2d_map=True)[::15 // fps]
        sequence = np.array(sequence) / ([4096, 6000] * (1 + player_count))

        game_state_gen = (
            (create_gamestates(space, player_count=4, ball_state=frame[:2], player_positions=frame[4:]), frame) for
        frame in
            sequence)

        images = list(create_image_stream(model, game_state_gen, space, arena, mask))
        # Save all images as a gif
        images[0].save(f'renderer_{_}.gif', save_all=True, append_images=images[1:], duration=seconds, loop=0)
