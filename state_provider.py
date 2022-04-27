from util import get_db, get_table_columns
con = get_db()

GAMEMODE_PLAYERS = {
  # gamemode_id, player_count
  'ssl_1v1': 2,
  'ssl_2v2': 4,
  'ssl_3v3': 6,
}

def get_replay_batch(gamemode, batch_size, verbose=False):
  if gamemode not in GAMEMODE_PLAYERS:
    raise ValueError('Invalid gamemode {}'.format(gamemode))
  cur = con.cursor()
  columns = get_table_columns(GAMEMODE_PLAYERS[gamemode])
  expression =f'''
  select {','.join(columns)} from {gamemode} where rowid in (
      select distinct (1+abs(random()) % (SELECT rowid FROM {gamemode} ORDER BY rowid DESC LIMIT 1)) from {gamemode} limit {batch_size}
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

def get_random_play_sequence(gamemode, batch_size, verbose=False):
  if gamemode not in GAMEMODE_PLAYERS:
    raise ValueError('Invalid gamemode {}'.format(gamemode))
  cur = con.cursor()
  columns = get_table_columns(GAMEMODE_PLAYERS[gamemode])
  expression =f'''
  select {','.join(columns)} from {gamemode} limit {batch_size} offset (
      select (1+abs(random()) % ((SELECT rowid FROM {gamemode} ORDER BY rowid DESC LIMIT 1) - {batch_size})) from {gamemode} limit 1
  );
  '''
  if verbose:
    print(expression)
  cur.execute(expression)
  return cur.fetchall()

def get_total_data_count(gamemode):
  """ Get the total number of rows in the table """
  if gamemode not in GAMEMODE_PLAYERS:
    raise ValueError('Invalid gamemode {}'.format(gamemode))
  cur = con.cursor()
  cur.execute(f'''select count(*) from {gamemode};''')
  return cur.fetchone()[0]

# Main test
if __name__ == '__main__':
  test_batch_size = 5000
  for gamemode in GAMEMODE_PLAYERS:
    batch, sequence = len(get_replay_batch(gamemode, 5000)), len(get_random_play_sequence(gamemode, 5000))
    assert test_batch_size == batch, 'Replay batch and test batch sizes do not match for {}, got: {}'.format(gamemode, batch)
    assert test_batch_size == sequence, 'Replay sequence and test batch sizes do not match for {}, got: {}'.format(gamemode, sequence)
    # Print the gamemode, the total data count for that game mode, and "OK" if the sizes match
    # Format the total data count with commas for thousands and spacing for 9 digits
    print('{} ({}): {}'.format(gamemode, '{:,}'.format(get_total_data_count(gamemode)), 'OK' if batch == sequence else 'FAIL'))