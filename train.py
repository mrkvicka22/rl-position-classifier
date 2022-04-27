from argparse import ArgumentParser
from collections import namedtuple
import random
import torch
from torch import save, load
from torch.optim import Adam, SGD
from torch.nn import BCEWithLogitsLoss, MSELoss
from net import create_model
from settings import TWOS_MODEL_PATH
import numpy as np

from state_provider import get_replay_batch, get_total_data_count

DatasetClass = namedtuple('DatasetClass', ['table', 'player_count'])
DATASET_SSL_1v1 = DatasetClass(table='ssl_1v1', player_count=2)
DATASET_SSL_2v2 = DatasetClass(table='ssl_2v2', player_count=4)
DATASET_SSL_3v3 = DatasetClass(table='ssl_3v3', player_count=6)


# TODO: Validate x and y coordinate space
# X is left and right
# Y is up and down

MODEL_PATH = TWOS_MODEL_PATH

def train_step(net, optimiser, loss_fn, inputs, targets):
  # optimiser.zero_grad()
  for param in net.parameters():
    param.grad = None
  y_pred = net(inputs)
  loss = loss_fn(y_pred, targets)
  loss.backward()
  optimiser.step()
  return loss.item()

def inversion(player_count):
  return np.array([-1, -1, 1] * (player_count + 1)) # players + ball

def normalization(player_count):
  return np.array([4096, 6000, 2000] * (player_count + 1)) # players + ball

def world_pos_to_map_pos(train_labels):
  return (train_labels * [4, 5] + [4, 5]).astype(np.int32)

def labels_to_map(labels):
  for x, y in world_pos_to_map_pos(labels):
    label = np.full((8, 10), 0.5, dtype=np.float32)

    # clamp to map size
    x = max(0, min(7, x))
    y = max(0, min(9, y))
    label[x, y] = 0
    yield label

RANDOM_POSITION_MUL = normalization(0) // [1, 1, 2]
RANDOM_POSITION_OFF = RANDOM_POSITION_MUL * [0, 0, 1] + [0, 0, 17] # cars drive at height 17

def get_state_batch(dataset, batch_size, suffix, random_position=False, augment_flip=False, augment_shuffle_blue=False, augment_shuffle_orange=False):
  batch = np.array(get_replay_batch(dataset.table, suffix, batch_size))
  if augment_flip and random.random() < 0.5: # flip teams
    batch *= inversion(dataset.player_count)

  # ball = batch[:, 0:3]
  # Get first team (blue), player_count includes both teams. Each player is 3 floats (x, y, z)
  first_team = batch[:, 3:3 + dataset.player_count * 3]
  # Get second team (orange)
  second_team = batch[:, 3 + dataset.player_count * 3:]

  # Randomly shuffle the first team
  if augment_shuffle_blue and random.random() < 0.5:
    np.random.shuffle(first_team)
  # Randomly shuffle the second team
  if augment_shuffle_orange and random.random() < 0.5:
    np.random.shuffle(second_team)

  # Produce labels which default to 1 (correct prediction) and get masked to 0 (incorrect prediction)
  labels = np.ones(batch_size)
  if random_position:
    # Create mask, which is the same size as the batch, and will mutable some of the batch and produce labels
    mask = np.zeros(batch_size, dtype=bool)
    # We will mask half the batch with random prediction positions
    mask[:batch_size // 2] = True
    # Shuffle the mask so that random parts of the batch are masked
    np.random.shuffle(mask)
    # Mask the prediction player (first blue player) with random a position
    # np.random.random produces floats from [0, 1), we transform this to [-1, 1) and then multiply by the normalization
    # such that the position is in the range of the map. We then add the offset to the height to make sure the cars
    # are in the field. (above ground)
    batch[mask, 3:6] = ((1 - 2 * np.random.random(size=(batch_size // 2, 3))) * RANDOM_POSITION_MUL + RANDOM_POSITION_OFF)  # (x, y, z) range.
    # Mask the label with 0 which is incorrect prediction
    labels[mask] = 0

  # input, label
  return batch / normalization(dataset.player_count), labels

def train(model, dataset: DatasetClass, epochs: int, batch_size: int, optimiser, loss_fn, random_position=False, augment_flip=False, augment_shuffle_blue=False, augment_shuffle_orange=False):

  epoch_length = get_total_data_count(dataset.table, 'train')
  print('Epoch length: ', epoch_length)
  total_steps = 0

  # Iterate for n epochs
  for epoch in range(epochs):
    # Count the total steps in the epoch
    epoch_steps = 0
    while epoch_steps < epoch_length:
      train_features, train_labels = get_state_batch(dataset, batch_size, 'train', random_position=random_position, augment_flip=augment_flip, augment_shuffle_blue=augment_shuffle_blue, augment_shuffle_orange=augment_shuffle_orange)
      inputs = torch.tensor(train_features.astype(np.float32))
      labels = torch.tensor(train_labels.astype(np.float32)).view((batch_size, 1)) # BCELoss requires strict size for labels
      train_step(model, optimiser, loss_fn, inputs, labels)

      epoch_steps += batch_size
      total_steps += batch_size
      if epoch_steps % 100_000 == 0:
        print('Epoch {}/{}, steps: {}, ({:.2f}%)'.format(epoch + 1, epochs, total_steps, epoch_steps / epoch_length * 100))

    model.eval()
    with torch.no_grad():
      # Validate
      val_features, val_labels = get_state_batch(dataset, batch_size, 'validation', random_position=random_position, augment_flip=False, augment_shuffle_blue=False, augment_shuffle_orange=False)
      val_inputs = torch.tensor(val_features.astype(np.float32))
      val_labels = torch.tensor(val_labels.astype(np.float32)).view((batch_size, 1)) # BCELoss requires strict size for labels
      val_loss = loss_fn(model(val_inputs), val_labels)
      print(f'Validation loss: {val_loss}')
      train_features, train_labels = get_state_batch(dataset, batch_size, 'train', random_position=random_position, augment_flip=False, augment_shuffle_blue=False, augment_shuffle_orange=False)
      train_inputs = torch.tensor(train_features.astype(np.float32))
      train_labels = torch.tensor(train_labels.astype(np.float32)).view((batch_size, 1))
      train_loss = loss_fn(model(train_inputs), train_labels)
      print(f'Training loss: {train_loss}')
      # Step scheduler
      # scheduler.step()
      print(f'Learning rate: {optimiser.param_groups[0]["lr"]}')
      # Save model
      # model_path = f'{model_path_base}_{steps_taken}.pt'
      # save(model, model_path)
      # print(f'Model saved to {model_path}')
      model.train()

if __name__ == '__main__':
  def __main__():

    available_datasets = [DATASET_SSL_1v1, DATASET_SSL_2v2, DATASET_SSL_3v3]

    # Create an argument parser for the command line arguments that
    # parses dataset (string), epochs (int), batch_size (int),
    # and model_path (str, optional)
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ssl_2v2')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100_000)
    # Add argument to specify optimiser and learning rate
    parser.add_argument('--optimiser', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    # Add argument to specify loss function
    parser.add_argument('--loss_fn', type=str, default='bce')
    # Add argument to specify rng seed
    parser.add_argument('--seed', type=int, default=1337)
    # Add arguments for augmentation
    parser.add_argument('--augment_flip', action='store_true')
    parser.add_argument('--augment_shuffle_blue', action='store_true')
    parser.add_argument('--augment_shuffle_orange', action='store_true')
    # Add argument to enable negative case (random position) generation
    # Provide help for this argument.
    parser.add_argument('--disable-rng-mask', action='store_true', help='Disables random position generation. Required for training with a negative mask, should prevent false positives when enabled.')

    # Parse all arguments into variables
    args = parser.parse_args()
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    optimiser = args.optimiser
    lr = args.lr
    loss_fn = args.loss_fn
    seed = args.seed
    augment_flip = args.augment_flip
    augment_shuffle_blue = args.augment_shuffle_blue
    augment_shuffle_orange = args.augment_shuffle_orange
    random_position = not args.disable_rng_mask

    # Apply seed to numpy, torch and python random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    # Validate that dataset is a valid dataset class table name
    dataset = dataset.lower()
    # Map dataset (dataset.table) to dataset class by iterating available_datasets
    for dataset_class in available_datasets:
      if dataset == dataset_class.table:
        dataset = dataset_class
        break

    # If dataset is not found, raise error
    if not isinstance(dataset, DatasetClass):
      valid_dataset_table_names = [dataset.table for dataset in available_datasets]
      raise ValueError(f"Dataset {dataset} not found, available datasets: {valid_dataset_table_names}")

    # Create model
    model = create_model(dataset.player_count)

    # Validate optimiser is adam or sgd, and create optimiser
    if optimiser == 'adam':
      optimiser = Adam(model.parameters(), lr=lr)
    elif optimiser == 'sgd':
      optimiser = SGD(model.parameters(), lr=lr)
    else:
      raise ValueError(f'Optimiser must be adam or sgd, not {optimiser}')

    # Validate loss function is cross_entropy or mse, and create loss function
    if loss_fn == 'bce':
      loss_fn = BCEWithLogitsLoss()
    elif loss_fn == 'mse':
      loss_fn = MSELoss()
    else:
      raise ValueError(f'Loss function must be bce or mse, not {loss_fn}')

    # Print summary of all arguments
    print("Beginning training...")
    print(f'Dataset: {dataset.table}, players: {dataset.player_count}')
    print(f'Dataset training size:')
    print(f'  Train     : {get_total_data_count(dataset.table, "train")}')
    print(f'  Validation: {get_total_data_count(dataset.table, "validation")}')
    print(f'  Test      : {get_total_data_count(dataset.table, "test")}')
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Optimiser: {optimiser}')
    # Print Learning rate in scientific notation
    print(f'Learning rate: {"{:2e}".format(lr)}')
    print(f'Loss function: {loss_fn}')
    print(f'Seed: {seed}')
    print(f'Augmentations:')
    print(f'  Flip teams: {augment_flip}')
    print(f'  Shuffle Blue: {augment_shuffle_blue}')
    print(f'  Shuffle Orange: {augment_shuffle_orange}')
    print(f'  Random position mask: {random_position}')
    print('\n')

    train(
      model
      , dataset
      , epochs # (total cycles through the dataset)
      , batch_size
      , optimiser
      , loss_fn
      , augment_flip
      , augment_shuffle_blue
      , augment_shuffle_orange
      , random_position
    )
  __main__()