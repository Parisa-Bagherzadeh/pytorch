import imp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS,N_LETTERS
from utils import load_data, letter_to_tensor , line_to_tensor, random_training_example



class RNN(nn.Module):
    def __init__(self,input_size , hidden_size , output_size) :
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size)
        
