import cv2 
import numpy as np

import torch
import torchvision

from argparse import ArgumentParser

from model import MyModel


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--device", type = str , default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)


    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0),(1))

    ])

    model = MyModel()
    model = model.to(device)
    model.load_state_dict(torch.load('weights.pth'))
    model.eval()

    image = cv2.imread('./sample_image.png')
    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image , (28,28))
    tensor = transform (image).unsqueeze(0).to(device)

    preds = model(tensor)
    preds = preds.cpu().detach().numpy()
    output = np.argmax(preds)

    classes = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneakers','Bag','Ankle Boot']

    print(classes[output])


