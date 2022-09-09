import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torchvision.utils import make_grid as grid_creation
from torch.utils.data import DataLoader

import numpy as np

import os

from utils import save_noise_images, plot_losses
from models import Generator, Critic_Discriminator

IMAGE_DIM = 64
NOISE_DIM = 100
BATCH_SIZE = 64
NUM_EPOCHS = 400
ALPHA = 5e-5
CLAMP_VALUE = 0.01
NUM_EPOCH_PER_GENERATOR_UPDATE = 10
NUM_EPOCH_PER_GENERATOR_TEST = 40

PLOT_LOSS = True
SAVE_NOISE_OUTPUT_IMAGES = True

PATH = "./"

transform = transforms.Compose([
            transforms.Resize((IMAGE_DIM,IMAGE_DIM)),
            transforms.CenterCrop(IMAGE_DIM),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
trainset = datasets.CIFAR10(PATH, train=True, download=True,
                                                       transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#testset = datasets.CIFAR10(root=PATH, train=False,
#                                       download=True, transform=transform)
#testloader = DataLoader(testset, batch_size=BATCH_SIZE,shuffle=False, num_workers=2)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


generator = Generator(NOISE_DIM)
if os.path.exists(PATH+'generator_cifar10_model.pth'):
    generator.load_state_dict(torch.load(PATH+'generator_cifar10_model.pth'))
    print("loaded last model!")

critic_discriminator = Critic_Discriminator(IMAGE_DIM)

generator.to(device)
critic_discriminator.to(device)

g_optim = torch.optim.RMSprop(generator.parameters(), lr=ALPHA)
c_optim = torch.optim.RMSprop(critic_discriminator.parameters(), lr=ALPHA)

test_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)

generator.train()
critic_discriminator.train()

g_loss_lst = []
c_loss_lst = []
output_noise_images = []

for epoch_num in range(1,NUM_EPOCHS+1):
    g_epoch_error = 0.0
    c_epoch_error = 0.0
    for batch_idx, data in enumerate(trainloader):
        batch_imgs, _ = data
        real_data = batch_imgs.to(device)
        train_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake_data = generator(train_noise)

        c_optim.zero_grad()
        total_critic_error = critic_discriminator(fake_data).mean() - critic_discriminator(real_data).mean()
        total_critic_error.backward()
        c_optim.step()
        for p in critic_discriminator.parameters(): 
            p.data.clamp_(-CLAMP_VALUE, CLAMP_VALUE) #LIPCHITZ FUNCTION F REQUIREMENT
        c_epoch_error += -1*total_critic_error #make it positive
        
        if (batch_idx+1)%NUM_EPOCH_PER_GENERATOR_UPDATE==0:
            fake_data = generator(torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device))
            g_optim.zero_grad()
            error_generator = -critic_discriminator(fake_data).mean()
            error_generator.backward()
            g_optim.step()
            g_epoch_error += error_generator

    print('Epoch {}: Critic_loss: {:.3f} Generator_loss: {:.3f}'.format(epoch_num, c_epoch_error, g_epoch_error))
    g_loss_lst.append(g_epoch_error.item())
    c_loss_lst.append(c_epoch_error.item())

    if epoch_num%NUM_EPOCH_PER_GENERATOR_TEST==0:
        fake_img = generator(test_noise).cpu().detach()
        output_noise_images.append(grid_creation(fake_img))


torch.save(generator.state_dict(), PATH+'generator_cifar10_model.pth')
print('Saved generator_cifar10_model')

if PLOT_LOSS == True:
    plot_losses(c_loss_lst,g_loss_lst)
    
if SAVE_NOISE_OUTPUT_IMAGES == True:
    save_noise_images(output_noise_images)
