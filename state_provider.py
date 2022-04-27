from util import get_db, get_table_columns
con = get_db()

GAMEMODE_PLAYERS = {
  # gamemode_id, player_count
  'ssl_1v1': 2,
  'ssl_2v2': 4,
  'ssl_3v3': 6,
}

DATA_TABLE_SUFFIXES = ['train', 'validation', 'test']

def get_replay_batch(gamemode, suffix, batch_size, verbose=False):
  if gamemode not in GAMEMODE_PLAYERS:
    raise ValueError('Invalid gamemode {}'.format(gamemode))
  if suffix not in DATA_TABLE_SUFFIXES:
    raise ValueError('Invalid suffix {}'.format(suffix))
  cur = con.cursor()
  columns = get_table_columns(GAMEMODE_PLAYERS[gamemode])
  expression =f'''
  select {','.join(columns)} from {gamemode}_{suffix} where rowid in (
      select distinct (1+abs(random()) % (SELECT rowid FROM {gamemode}_{suffix} ORDER BY rowid DESC LIMIT 1)) from {gamemode}_{suffix} limit {batch_size}
  );
  '''
  if verbose:
    print(expression)
  cur.execute(expression)
  results = cur.fetchall()
  if 0 < len(results) < batch_size:
    print('WARNING: Batch size {} is larger than the number of rows ({}) in the table {}, making multiple requests.'.format(batch_size, len(results), gamemode))
    results += get_replay_batch(gamemode, batch_size - len(results))
  return results

def get_random_play_sequence(gamemode, suffix, batch_size, verbose=False):
  if gamemode not in GAMEMODE_PLAYERS:
    raise ValueError('Invalid gamemode {}'.format(gamemode))
  if suffix not in DATA_TABLE_SUFFIXES:
    raise ValueError('Invalid suffix {}'.format(suffix))
  cur = con.cursor()
  columns = get_table_columns(GAMEMODE_PLAYERS[gamemode])
  expression =f'''
  select {','.join(columns)} from {gamemode}_{suffix} limit {batch_size} offset (
      select (1+abs(random()) % ((SELECT rowid FROM {gamemode}_{suffix} ORDER BY rowid DESC LIMIT 1) - {batch_size})) from {gamemode}_{suffix} limit 1
  );
  '''
  if verbose:
    print(expression)
  cur.execute(expression)
  return cur.fetchall()

def get_total_data_count(gamemode, suffix):
  """ Get the total number of rows in the table """
  if gamemode not in GAMEMODE_PLAYERS:
    raise ValueError('Invalid gamemode {}'.format(gamemode))
  if suffix not in DATA_TABLE_SUFFIXES:
    raise ValueError('Invalid suffix {}'.format(suffix))
  cur = con.cursor()
  cur.execute(f'''select count(*) from {gamemode}_{suffix};''')
  return cur.fetchone()[0]

# Main test
if __name__ == '__main__':
  test_batch_size = 5000
  for gamemode in GAMEMODE_PLAYERS:
    for suffix in DATA_TABLE_SUFFIXES:
      batch, sequence = len(get_replay_batch(gamemode, suffix, 5000)), len(get_random_play_sequence(gamemode, suffix, 5000))
      assert test_batch_size == batch, 'Replay batch and test batch sizes do not match for {}_{}, got: {}'.format(gamemode, suffix, batch)
      assert test_batch_size == sequence, 'Replay sequence and test batch sizes do not match for {}_{}, got: {}'.format(gamemode, suffix, sequence)
      # Print the gamemode, the total data count for that game mode, and "OK" if the sizes match
      # Format the total data count with commas for thousands and spacing for 9 digits
      print('{}_{} ({}): {}'.format(gamemode, suffix, '{:,}'.format(get_total_data_count(gamemode, suffix)), 'OK' if batch == sequence else 'FAIL'))