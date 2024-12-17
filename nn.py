import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Define neural network that I will train using coarse data samples 
#Use this as an initial guess for the fine scale training
torch.set_default_dtype(torch.float64)
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # torch.set_default_dtype(torch.float64)

        self.flatten = nn.Flatten()
        #Input layer is 2 dimensional because I have (x,y) information in data 
        #Output layer is 1 dimensional because I want to output the temperature
        #at that particular point 
        
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(2,20),
            nn.Tanh(),
            nn.Linear(20,20),
            nn.Tanh(),
            nn.Linear(20,15),
            nn.Tanh(),
            nn.Linear(15,10),
            nn.Tanh(),
            nn.Linear(10,1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_tanh_stack(x)
        return logits


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

#Show data types of model parameters 
for param in model.linear_tanh_stack.parameters():
  print(param.data.dtype)

