import cv2
from matplotlib import test 
import numpy as np
from sklearn.utils import shuffle

import torch
import torchvision

from argparse import ArgumentParser

from model import MyModel


if __name__ == '__main__':

    batch_size = 64
    test_acc = 0.0

    parser = ArgumentParser()
    parser.add_argument("--device", type = str ,default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)

    

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0),(1))

    ])

    test_data = torchvision.datasets.FashionMNIST('./test_data' , train = False , download = True , transform = transform)
    test_data_loader = torch.utils.data.DataLoader(test_data , batch_size = batch_size , shuffle = True)

    model = MyModel()
    model = model.to(device)
    model.load_state_dict(torch.load('weights.pth'))
    model.eval()


    for images , labels in test_data_loader :
        
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        test_acc += model.calc_acc(preds , labels)

    total_acc = test_acc / len(test_data_loader)  

    print(f'Test Accuracy : {total_acc}')  

    
    

