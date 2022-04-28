import train
import numpy as np
import torch

# Create two loss functions, MSE and BCE
criterion = torch.nn.MSELoss()
criterion2 = torch.nn.BCEWithLogitsLoss()

model = torch.load('model_ssl_2v2_latest.pt')

batch_size = min(50_000, train.get_total_data_count('ssl_2v2', 'train'))
mse, bce, acc = 0, 0, 0
for _ in range(10):
  init_features, init_labels = train.get_state_batch(train.DATASET_SSL_2v2, batch_size, 'train', random_position=_ % 2 == 0, augment_flip=False, use_2d_map=False)
  init_inputs = torch.tensor(init_features.astype(np.float32))
  init_labels = torch.tensor(init_labels.astype(np.float32)).view((batch_size, 1))
  mse += criterion(model(init_inputs), init_labels).item()
  bce += criterion2(model(init_inputs), init_labels).item()
  # Compute accuracy
  acc += (model(init_inputs) > 0.5).sum().item() / batch_size

print(f'MSE: {mse/10}, BCE: {bce/10}, Accuracy: {acc/10}')
