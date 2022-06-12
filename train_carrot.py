import time

import torch
from torch import optim
from torch import nn
from dataset import SSLDataset
from nets import GeneratorNet, DiscriminatorNet
from torch.autograd import Variable

"""
TABLES:
tables = cursor.execute('SELECT name from sqlite_master where type="table"')
- 'ssl_2v2_train'
- 'sqlite_sequence'
- 'ssl_2v2_validation'
- 'ssl_2v2_test'

"""


def initialize_datasets(device, batchsize):
    """Return datasets in order train, val, test"""
    print("Started dataset init")
    start = time.perf_counter()
    train = SSLDataset('clean_ssl_2v2_train', device, batchsize)
    val = SSLDataset('clean_ssl_2v2_validation', device, batchsize)
    test = SSLDataset('clean_ssl_2v2_test', device, batchsize)
    print("Dataset init time:", time.perf_counter())
    return train, val, test


def train(config):
    train_dataset, val_dataset, test_dataset = initialize_datasets(config["device"], config["batch_size"])
    generator_net = GeneratorNet().to(device=config["device"])
    generator_optimizer = optim.Adam(generator_net.parameters(), lr=config["generator_lr"])
    discriminator_net = DiscriminatorNet().to(device=config["device"])
    discriminator_optimizer = optim.Adam(discriminator_net.parameters(), lr=config["discriminator_lr"])
    loss_f = nn.CrossEntropyLoss()
    print("Setup done")
    # print(f"Dataset length: {len(train_dataset)}")
    for epoch in range(config["epochs"]):

        for i in range(300):
            for _ in range(config["n_critic"]):
                # Sample data
                z = Variable(torch.randn((config["batch_size"], 73),device=config["device"]))
                X = Variable(train_dataset.get_batch())

                # Dicriminator forward-loss-backward-update
                G_sample = generator_net(z)
                D_real = discriminator_net(X)
                D_fake = discriminator_net(G_sample)

                D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

                D_loss.backward()
                discriminator_optimizer.step()

                # Weight clipping
                for p in discriminator_net.parameters():
                    p.data.clamp_(-config["clamp"], config["clamp"])

                # Housekeeping - reset gradient
                generator_net.zero_grad()
                discriminator_net.zero_grad()

                # Generator forward-loss-backward-update
            z = Variable(torch.randn((config["batch_size"], 73),device=config["device"]))

            G_sample = generator_net(z)
            D_fake = discriminator_net(G_sample)

            G_loss = -torch.mean(D_fake)

            G_loss.backward()
            generator_optimizer.step()

            # Housekeeping - reset gradient
            generator_net.zero_grad()
            discriminator_net.zero_grad()
            print(
                f"Epoch: {epoch + 1} Batch: {i + 1} \nDiscriminator loss: {D_loss.item()}\nGenerator loss: {G_loss.item()}")
        torch.save({
            'Generator_state_dict': generator_net.state_dict(),
            'Discriminator_state_dict': discriminator_net.state_dict()
        }, f"GAN_models_{epoch}.pt")


if __name__ == '__main__':
    config = {"batch_size": 1024,
              "generator_lr": 3e-4,
              "discriminator_lr": 3e-4,
              "epochs": 15,
              "clamp":0.01,
              "n_critic":1,
              "device": torch.device("cuda")}
    train(config)
