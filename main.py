#Importing the dependencies
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger
from generator import Generator
from discriminator import Discriminator

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

gen = Generator()
dis = Discriminator()

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    n = Variable(torch.randn(size, 100))
    return n

d_optimizer = optim.Adam(dis.parameters(), lr=0.0002)
g_optimizer = optim.Adam(gen.parameters(), lr=0.0002)

loss = nn.BCELoss()

def ones_target(size):
    data = Variable(torch.ones(size,1))
    return data

def zeros_target(size):
    data = Variable(torch.zeros(size,1))
    return data

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()

    #Train on real data
    prediction_real = dis(real_data)
    error_real = loss(prediction_real, zeros_target(N))
    error_real.backward()

    #Train on fake data
    prediction_fake = dis(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    #Update the weights with gradients
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)

    #reseting the gradients
    optimizer.zero_grad()

    prediction = dis(fake_data)
    error = loss(prediction, ones_target(N))
    error.backward()
    optimizer.step()
    return error

num_test_samples = 16
test_noice = noise(num_test_samples)

logger = Logger(model_name='VGAN', data_name='MNIST')

num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        
        """
        Training the disctiminator
        """
        real_data = Variable(images_to_vectors(real_batch))

        #Generate fake data
        fake_data = gen(noise(N)).detach()

        d_error, d_pred_real, d_pred_fake = \
            train_discriminator(d_optimizer, real_data, fake_data)
        
        """
        Training the generator
        """
        fake_data = gen(noise(N))

        g_error = train_generator(g_optimizer, fake_data)

        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        if(n_batch)%100 == 0:
            test_images = vectors_to_images(gen(test_noice))
            test_images = test_images.data   

            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            )
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
