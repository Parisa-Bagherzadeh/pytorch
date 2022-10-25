import torch

from sklearn.datasets import make_blobs



class mlp(torch.nn.Module):
    def __init__(self , input_size , hidden_size) :
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output





        