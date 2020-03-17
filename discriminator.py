#Importing the dependencies
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

class Discriminator(torch.nn.Module):
    """
    A 3 hidden layer disctimative neural network
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hid0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hid1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hid2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hid0(x)
        x = self.hid1(x)
        x = self.hid2(x)
        x = self.out(x)
        return x