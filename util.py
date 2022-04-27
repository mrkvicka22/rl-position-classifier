
from itertools import chain
import sqlite3

def get_table_columns(player_count, use_2d_map=False):
  if use_2d_map:
    player_columns = chain(*([f'player_{1 + i}_x', f'player_{1 + i}_y'] for i in range(player_count)))
    return ['ball_x', 'ball_y'] + list(player_columns)
  else:
    player_columns = chain(*([f'player_{1 + i}_x', f'player_{1 + i}_y', f'player_{1 + i}_z'] for i in range(player_count)))
    return ['ball_x', 'ball_y', 'ball_z'] + list(player_columns)

def get_db():
  return sqlite3.connect('C:\\Users\\jneve\\Desktop\\replays.db')