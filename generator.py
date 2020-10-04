import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

class Generator(torch.nn.Module):
    """
    A 3 hidden layer generative neural network
    """
    def __init__(self):
        super(Generator, self).__init__() #super() makes it superclass
        n_features = 100
        n_out = 784

        self.hid0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hid1 = nn.Sequential(
            nn.Linear(256,512),
            nn.LeakyReLU(0.2)
        )
        self.hid2 = nn.Sequential(
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
            nn.Linear(1024,n_out)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hid0(x)
        x = self.hid1(x)
        x = self.hid2(x)
        x = self.out(x)
        return x
