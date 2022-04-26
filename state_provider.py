from util import get_db, get_table_columns
con = get_db()

GAMEMODE_PLAYERS = {
  # gamemode_id, player_count
  'ssl_1v1': 2,
  'ssl_2v2': 4,
  'ssl_3v3': 6,
}

def get_replay_batch(gamemode, batch_size):
  if gamemode not in GAMEMODE_PLAYERS:
    raise ValueError('Invalid gamemode {}'.format(gamemode))
  cur = con.cursor()
  columns = get_table_columns(GAMEMODE_PLAYERS[gamemode])
  cur.execute(f'''
  select {','.join(columns)} from {gamemode} where rowid in (
      select (1+abs(random()) % (SELECT rowid FROM {gamemode} ORDER BY rowid DESC LIMIT 1)) from {gamemode} limit {batch_size + 100}
  ) limit {batch_size};
  ''')
  results = cur.fetchall()
  if len(results) < batch_size:
    # print('WARNING: Batch size {} is larger than the number of rows in the table {}'.format(batch_size, gamemode))
    results += get_replay_batch(gamemode, batch_size - len(results))
  return results

def get_random_play_sequence(gamemode, batch_size):
  if gamemode not in GAMEMODE_PLAYERS:
    raise ValueError('Invalid gamemode {}'.format(gamemode))
  cur = con.cursor()
  columns = get_table_columns(GAMEMODE_PLAYERS[gamemode])
  cur.execute(f'''
  select {','.join(columns)} from {gamemode} limit {batch_size} offset (
      select (1+abs(random()) % ((SELECT rowid FROM {gamemode} ORDER BY rowid DESC LIMIT 1) - {batch_size})) from {gamemode} limit 1
  );
  ''')
  return cur.fetchall()

# Main test
if __name__ == '__main__':
  for gamemode in GAMEMODE_PLAYERS:
    print(gamemode)
    print(len(get_replay_batch(gamemode, 5000)))