import itertools
import os
import random
from argparse import ArgumentParser
from collections import namedtuple
import traceback

import numpy as np
import torch
import wandb
from torch import load, save
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import SGD, Adam

from net import create_model
from state_provider import get_replay_batch, get_total_data_count

from renderer import create_animation_from_model

DatasetClass = namedtuple('DatasetClass', ['table', 'player_count'])
DATASET_SSL_1v1 = DatasetClass(table='ssl_1v1', player_count=2)
DATASET_SSL_2v2 = DatasetClass(table='ssl_2v2', player_count=4)
DATASET_SSL_3v3 = DatasetClass(table='ssl_3v3', player_count=6)

# TODO: Validate x and y coordinate space
# X is left and right
# Y is up and down

def train_step(net, optimiser, loss_fn, inputs, targets):
  # optimiser.zero_grad()
  for param in net.parameters():
    param.grad = None
  y_pred = net(inputs)
  loss = loss_fn(y_pred, targets)
  loss.backward()
  optimiser.step()
  return loss.item()

def inversion(player_count, use_2d_map=False, x_inversion=True):
  inversion_map = [-1, 1] if x_inversion else [1, -1]
  # players + ball
  if use_2d_map:
    return np.array(inversion_map * (player_count + 1))
  inversion_map += [1]
  return np.array(inversion_map * (player_count + 1))

def normalization(player_count, use_2d_map=False):
  if use_2d_map:
    return np.array([4096, 6000] * (player_count + 1))
  return np.array([4096, 6000, 2000] * (player_count + 1)) # players + ball

def create_random_mask(batch_size):
  # Create mask, which is the same size as the batch, that can be used to mutate the batch data
  mask = np.zeros(batch_size, dtype=bool)
  # We will mask half the batch so that 50% of the batch is mutated
  mask[:batch_size // 2] = True
  np.random.shuffle(mask)
  return mask

def apply_normal_randomization(batch, use_2d_map=False):
  batch_size = batch.shape[0]
  # The amount of dimensions on each entity, 2 is (x, y), 3 is (x, y, z)
  ndims = 2 if use_2d_map else 3
  # Produce labels which default to 1 (correct prediction) and get masked to 0 (incorrect prediction)
  labels = np.ones(batch_size)
  rng_pos_mul = normalization(0, use_2d_map=use_2d_map)
  if not use_2d_map:
    rng_pos_mul //= [1, 1, 2]
    rng_pos_off = rng_pos_mul * [0, 0, 1] + [0, 0, 17] # cars drive at height 17
  else:
    rng_pos_off = [0, 0]

  mask = create_random_mask(batch_size)
  # Mask the prediction player (first blue player) with random a position
  # np.random.random produces floats from [0, 1), we transform this to [-1, 1) and then multiply by the normalization
  # such that the position is in the range of the map. We then add the offset to the height to make sure the cars
  # are in the field. (above ground)
  batch[mask, ndims:2*ndims] = ((1 - 2 * np.random.random(size=(batch_size // 2, ndims))) * rng_pos_mul + rng_pos_off)  # (x, y, z) range.
  # Mask the label with 0 which is incorrect prediction
  labels[mask] = 0
  return labels

def apply_shuffle_randomization(batch, use_2d_map=False):
  batch_size = batch.shape[0]
  # The amount of dimensions on each entity, 2 is (x, y), 3 is (x, y, z)
  ndims = 2 if use_2d_map else 3
  quarter_size = batch_size // 4
  third_quarter, fourth_quarter = batch[-quarter_size*2:-quarter_size], batch[-quarter_size:]
  # Swap the third predictions with the fourth predictions
  # Perform a copy because views can't be swapped all fun like, or they can but it's hard and I'm tired.
  third_predictions = third_quarter[:, ndims:2*ndims].copy()
  fourth_predictions = fourth_quarter[:, ndims:2*ndims]
  third_quarter[:, ndims:2*ndims] = fourth_predictions
  fourth_quarter[:, ndims:2*ndims] = third_predictions
  # Create labels with first half of ones and second half of zeros
  labels = np.ones(batch_size)
  labels[-quarter_size*2:] = 0
  # Shuffle the batch and labels in unison
  # This rng state getter/setter makes everyone feel uneasy, so consider changing to arange or something
  # Ideally we keep the modify in place.
  rng_state = np.random.get_state()
  np.random.shuffle(batch)
  np.random.set_state(rng_state)
  np.random.shuffle(labels)
  return labels

def get_state_batch(dataset, batch_size, suffix, randomisation=None, augment_flip=False, use_2d_map=False):
  batch = np.array(get_replay_batch(dataset.table, suffix, batch_size, use_2d_map=use_2d_map))

  # Apply augmentations, flip teams
  if augment_flip:
    x_inversion_mask = create_random_mask(batch_size)
    # flip on x axis
    batch[x_inversion_mask] *= inversion(dataset.player_count, use_2d_map=use_2d_map, x_inversion=True)

  # Apply randomization
  if randomisation == 'normal':
    labels = apply_normal_randomization(batch, use_2d_map=use_2d_map)
  elif randomisation == 'shuffle':
    labels = apply_shuffle_randomization(batch)
  else:
    labels = np.ones(batch_size)

  # input, label
  return batch / normalization(dataset.player_count, use_2d_map=use_2d_map), labels

def train(model, dataset: DatasetClass, epochs: int, batch_size: int, optimiser, loss_fn, randomisation=None, augment_flip=False, use_2d_map=False):

  # TODO: The train, validation, periodic test, and final evaluation all are very similar
  #       We should probably abstract this out into a function.

  epoch_length = get_total_data_count(dataset.table, 'train')
  print('Epoch length: ', epoch_length)
  total_steps = 0
  init_features, init_labels = get_state_batch(dataset, batch_size, 'train', randomisation=randomisation, augment_flip=False, use_2d_map=use_2d_map)
  init_inputs = torch.tensor(init_features.astype(np.float32))
  init_labels = torch.tensor(init_labels.astype(np.float32)).view((batch_size, 1))
  init_loss = loss_fn(model(init_inputs), init_labels).item()
  print(f'Init loss: {init_loss}')

  best_val_loss, best_val_acc = init_loss, 0

  epoch_iterator = range(epochs) if epochs is not None else itertools.count()
  epoch_total = f"/{epochs}" if epochs is not None else ""

  # Iterate for n epochs, or forever if epochs is None
  for epoch in epoch_iterator:
    # Count the total steps in the epoch
    epoch_steps = 0
    while epoch_steps < epoch_length:
      train_features, train_labels = get_state_batch(dataset, batch_size, 'train', randomisation=randomisation, augment_flip=augment_flip, use_2d_map=use_2d_map)
      inputs = torch.tensor(train_features.astype(np.float32))
      labels = torch.tensor(train_labels.astype(np.float32)).view((batch_size, 1)) # BCELoss requires strict size for labels
      training_loss = train_step(model, optimiser, loss_fn, inputs, labels)

      epoch_steps += batch_size
      total_steps += batch_size
      if epoch_steps % 100_000 == 0:
        print('Epoch {}{}, steps: {}, ({:.2f}%)'.format(epoch + 1, epoch_total, total_steps, epoch_steps / epoch_length * 100))
      wandb.log({'train_loss': training_loss, 'learning_rate': optimiser.param_groups[0]["lr"], 'epoch': epoch, 'steps': total_steps}, commit=False)

    model.eval()
    with torch.no_grad():
      validation_batch_size = 50_000
      # Validate
      val_features, val_labels = get_state_batch(dataset, validation_batch_size, 'validation', randomisation=randomisation, augment_flip=False, use_2d_map=use_2d_map)
      val_inputs = torch.tensor(val_features.astype(np.float32))
      val_labels = torch.tensor(val_labels.astype(np.float32)).view((validation_batch_size, 1)) # BCELoss requires strict size for labels
      val_pred = model(val_inputs)
      val_pred_threshold = 0 if isinstance(loss_fn, BCEWithLogitsLoss) else 0.5
      val_loss = loss_fn(val_pred, val_labels).item()
      val_acc = (val_labels == (val_pred > val_pred_threshold)).float().mean().item()
      # Update best model
      if val_acc > best_val_acc:
        save(model, os.path.join(wandb.run.dir, f"model_{dataset.table}_best.pt"))
      # Save best loss and best accuracy
      wandb.run.summary["best_val_loss"] = best_val_loss = min(val_loss, best_val_loss)
      wandb.run.summary["best_val_acc"] = best_val_acc = max(val_acc, best_val_acc)
      print(f'Validation loss: {val_loss}, accuracy: {val_acc}')
      train_features, train_labels = get_state_batch(dataset, validation_batch_size, 'train', randomisation=randomisation, augment_flip=False, use_2d_map=use_2d_map)
      train_inputs = torch.tensor(train_features.astype(np.float32))
      train_labels = torch.tensor(train_labels.astype(np.float32)).view((validation_batch_size, 1))
      train_loss = loss_fn(model(train_inputs), train_labels)
      print(f'Training loss: {train_loss}')
      # Step scheduler
      # scheduler.step()
      # print(f'Learning rate: {optimiser.param_groups[0]["lr"]}')
      # Save model
      save(model, os.path.join(wandb.run.dir, f"model_{dataset.table}_latest.pt"))
      save(model, os.path.join(wandb.run.dir, f"model_{dataset.table}_chk_{total_steps}.pt"))
      wandb.log({'val_loss': val_loss, 'val_acc': val_acc, 'train_loss': train_loss, 'learning_rate': optimiser.param_groups[0]["lr"], 'epoch': epoch, 'steps': total_steps})
      model.train()

    if epochs is not None and epochs % 10 == 0:
      print("Performing periodic test evaluation on forever run.")
      try:
        model.eval()
        with torch.no_grad():
          test_batch_size, test_passes = 100_000, 10
          # Multiple passes, and average the result
          test_loss, test_acc = 0, 0
          for _ in range(test_passes):
            test_features, test_labels = get_state_batch(dataset, test_batch_size, 'test', randomisation=randomisation, augment_flip=False, use_2d_map=use_2d_map)
            test_inputs = torch.tensor(test_features.astype(np.float32))
            test_labels = torch.tensor(test_labels.astype(np.float32)).view((test_batch_size, 1)) # BCELoss requires strict size for labels
            test_pred = model(test_inputs)
            test_pred_threshold = 0 if isinstance(loss_fn, BCEWithLogitsLoss) else 0.5
            test_loss += loss_fn(test_pred, test_labels).item()
            test_acc += (test_labels == (test_pred > test_pred_threshold)).float().mean().item()
          test_loss /= test_passes
          test_acc /= test_passes
          print(f'Test loss: {test_loss}, accuracy: {test_acc}')
          wandb.log({'test_loss': test_loss, 'test_acc': test_acc, 'epoch': epoch, 'steps': total_steps})
        print("Rendering best model video...")
        best_model_path = os.path.join(wandb.run.dir, f"model_{dataset.table}_best.pt")
        # Check if the best model exists
        if not os.path.exists(best_model_path):
          print("Best model not found, skipping video rendering")
          return
        # Create a dir called 'renderoutput'
        render_dir = os.path.join(wandb.run.dir, "renderoutput")
        if not os.path.exists(render_dir):
          os.mkdir(render_dir)
        # Get a path to render dir renderoutput.gif
        render_path = os.path.join(render_dir, "output.gif")
        # Render
        create_animation_from_model(best_model_path, render_path, player_count=dataset.player_count, image_size=0.5)
        wandb.log({"video": wandb.Video(render_path, fps=4, format="gif")})
      except:
        print("Error performing test evaluation and rendering video, skipping")
        traceback.print_exc()

  # Calculate the final loss and accuracy
  print("Running final evaluation against test data...")
  model.eval()
  with torch.no_grad():
    test_batch_size, test_passes = 100_000, 10
    # Multiple passes, and average the result
    test_loss, test_acc = 0, 0
    for _ in range(test_passes):
      test_features, test_labels = get_state_batch(dataset, test_batch_size, 'test', randomisation=randomisation, augment_flip=False, use_2d_map=use_2d_map)
      test_inputs = torch.tensor(test_features.astype(np.float32))
      test_labels = torch.tensor(test_labels.astype(np.float32)).view((test_batch_size, 1)) # BCELoss requires strict size for labels
      test_pred = model(test_inputs)
      test_pred_threshold = 0 if isinstance(loss_fn, BCEWithLogitsLoss) else 0.5
      test_loss += loss_fn(test_pred, test_labels).item()
      test_acc += (test_labels == (test_pred > test_pred_threshold)).float().mean().item()
    test_loss /= test_passes
    test_acc /= test_passes
    print(f'Test loss: {test_loss}, accuracy: {test_acc}')
    wandb.run.summary["test_loss"] = test_loss
    wandb.run.summary["test_acc"] = test_acc
  print("Rendering best model video...")
  best_model_path = os.path.join(wandb.run.dir, f"model_{dataset.table}_best.pt")
  # Check if the best model exists
  if not os.path.exists(best_model_path):
    print("Best model not found, skipping video rendering")
    return
  # Create a dir called 'renderoutput'
  render_dir = os.path.join(wandb.run.dir, "renderoutput")
  if not os.path.exists(render_dir):
    os.mkdir(render_dir)
  # Get a path to render dir renderoutput.gif
  render_path = os.path.join(render_dir, "output.gif")
  # Render
  create_animation_from_model(best_model_path, render_path, player_count=dataset.player_count, image_size=0.5)
  wandb.log({"video": wandb.Video(render_path, fps=4, format="gif")})


if __name__ == '__main__':
  def __main__():

    available_datasets = [DATASET_SSL_1v1, DATASET_SSL_2v2, DATASET_SSL_3v3]

    # Create an argument parser for the command line arguments that
    # parses dataset (string), epochs (int), batch_size (int),
    # and model_path (str, optional)
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ssl_2v2', choices=[d.table for d in available_datasets], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--bs', type=int, default=500, help='Batch size')
    # Add argument to specify optimiser and learning rate
    parser.add_argument('--optimiser', type=str, default='adam', choices=['adam', 'sgd'], help='Optimiser to use')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    # Add argument to specify loss function
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'mse'], help='Loss function to use')
    # Add argument to specify rng seed
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    # Add arguments for augmentation
    parser.add_argument('--aug-flip', action='store_true', help="Flip blue and orange teams 50%% of the time")
    # Add argument to choose randomisation strategy, options are normal, shuffle, or none. Defaults to normal
    parser.add_argument('--randomisation', type=str, default='normal', choices=['normal', 'shuffle', 'none'], help='Random position strategy')
    # Add argument to use a 2d map, use the variable "use_2d_map" in the code
    parser.add_argument('--2d', action='store_true', dest='use_2d_map', help='Use a 2d map')
    # Add an argument to specify the amount of hidden layers
    parser.add_argument('--hidden-layers', type=int, default=2, help='Number of hidden layers')
    # Add an argument to specify the amount of hidden units
    parser.add_argument('--hidden-units', type=int, default=128, help='Number of hidden units')
    # Add an optional argument to specify dropout percentage, defaults to None
    parser.add_argument('--dropout', type=float, default=None, help='Dropout percentage, defaults to None which disables dropout. Range 0-1.')
    # Add an argument to specify the dropout layers, defaults to the hidden layer count, ignored if dropout percentage is not specified
    parser.add_argument('--dropout-layers', type=int, default=None, help='Number of dropout layers, defaults to number of hidden layers, ignored if dropout percentage is not specified')
    # Add a boolean argument to specify if the training should run forever
    parser.add_argument('--forever', action='store_true', help='Train forever')

    # Parse all arguments, ship to wandb and collect the returned config
    args = parser.parse_args()
    wandb.init(project="rl-position-classifier", entity='nevercast', config=args)
    args = wandb.config

    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.bs
    optimiser = args.optimiser
    lr = args.lr
    loss_fn = args.loss
    seed = args.seed
    augment_flip = args.aug_flip
    randomisation = args.randomisation.lower()
    randomisation = None if randomisation == 'none' else randomisation
    use_2d_map = args.use_2d_map
    hidden_layers = args.hidden_layers
    hidden_units = args.hidden_units
    dropout = args.dropout
    dropout_layers = args.dropout_layers
    forever = args.forever
    if forever:
      epochs = None

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

    # Create model and attach wandb
    model = create_model(dataset.player_count, 2 if use_2d_map else 3, hidden_layers=hidden_layers, hidden_units=hidden_units, dropout=dropout, dropout_layers=dropout_layers)
    wandb.watch(model)
    wandb.save(os.path.join(wandb.run.dir, f"model_{dataset.table}_chk*"))

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
    print(f'Epochs: {epochs} {("(forever)" if forever else "")}')
    print(f'Batch size: {batch_size}')
    print(f'Optimiser: {optimiser}')
    # Print Learning rate in scientific notation
    print(f'Learning rate: {"{:2e}".format(lr)}')
    print(f'Loss function: {loss_fn}')
    print(f'Seed: {seed}')
    print(f'Augmentations:')
    print(f'  Flip teams: {augment_flip}')
    print(f'  Position randomisation: {randomisation}')
    print(f'Use 2D map: {use_2d_map}')
    print(f'Hidden layers: {hidden_layers}')
    print(f'Hidden units: {hidden_units}')
    print(f'Dropout: {dropout}')
    print(f'Dropout layers: {dropout_layers}')
    print('\n')

    train(
      model=model
      , dataset=dataset
      , epochs=epochs # (total cycles through the dataset)
      , batch_size=batch_size
      , optimiser=optimiser
      , loss_fn=loss_fn
      , randomisation=randomisation
      , augment_flip=augment_flip
      , use_2d_map=use_2d_map
    )
  __main__()
