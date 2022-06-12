import time

import torch
from torch import optim
from torch import nn
from dataset import SSLDataset
from nets import GeneratorNet, DiscriminatorNet

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
    val = SSLDataset('clean_ssl_2v2_validation', device,batchsize)
    test = SSLDataset('clean_ssl_2v2_test', device,batchsize)
    print("Dataset init time:", time.perf_counter())
    return train,val,test


def train(config):
    train_dataset, val_dataset, test_dataset = initialize_datasets(config["device"], config["batch_size"])
    generator_net = GeneratorNet().to(device=config["device"])
    generator_optimizer = optim.Adam(generator_net.parameters(),lr=config["generator_lr"])
    discriminator_net = DiscriminatorNet().to(device=config["device"])
    discriminator_optimizer = optim.Adam(discriminator_net.parameters(),lr=config["discriminator_lr"])
    loss_f = nn.CrossEntropyLoss()
    print("Setup done")
    #print(f"Dataset length: {len(train_dataset)}")
    for epoch in range(config["epochs"]):

        for i in range(300):
            # get real data
            start_batch = time.perf_counter()
            true_input = train_dataset.get_batch()
            print("ended batch", time.perf_counter() - start_batch)

            start_ml = time.perf_counter()
            # zero the parameter gradients
            generator_optimizer.zero_grad()

            # Get inputs
            print("Inference started")
            noise = torch.rand((config["batch_size"],73),device=config["device"])
            generated_input = generator_net(noise)
            print("started")

            # Train generator
            generator_discriminator_out = discriminator_net(generated_input)
            generator_loss = loss_f(generator_discriminator_out, torch.ones(config["batch_size"],device=config["device"],dtype=torch.long))
            generator_loss.backward()
            generator_optimizer.step()

            # Train discriminator
            discriminator_optimizer.zero_grad()
            true_discriminator_out = discriminator_net(true_input)
            print(len(true_input))
            true_discriminator_loss = loss_f(true_discriminator_out, torch.ones(len(true_input),device=config["device"],dtype=torch.long))

            generator_discriminator_out = discriminator_net(generated_input.detach())
            generator_discriminator_loss = loss_f(generator_discriminator_out, torch.zeros(config["batch_size"], device=config["device"],dtype=torch.long))
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()
            print(f"Epoch: {epoch+1} Batch: {i+1} Time:{time.perf_counter()-start_ml}\nDiscriminator loss: {discriminator_loss.item()}\nGenerator loss: {generator_loss.item()}")
        torch.save({
            'Generator_state_dict': generator_net.state_dict(),
            'Discriminator_state_dict': discriminator_net.state_dict()
        }, f"GAN_models_{epoch}.pt")

if __name__ == '__main__':
    config = {"batch_size":100_000,
              "generator_lr":3e-4,
              "discriminator_lr":3e-4,
              "epochs":15,
              "device":torch.device("cuda")}
    train(config)


