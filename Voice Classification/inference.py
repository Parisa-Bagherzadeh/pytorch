from model import M5

import torch
import torchaudio

import numpy as np

from argparse import ArgumentParser



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()
    parser.add_argument("--voice", type = str )
    args = parser.parse_args()
    
    signal, sample_rate = torchaudio.load(args.voice)


    model = M5(n_output = 11).to(device)
    model = model.to(device)
    model.load_state_dict(torch.load('weights.pth'))

    model.eval()


    signal = torch.mean(signal, dim = 0, keepdim = True)
    transform = torchaudio.transforms.Resample(sample_rate, 8000)
    signal = transform(signal)

    tensor = signal.unsqueeze(0).to(device)

    preds = model(tensor)

    preds = preds.cpu().detach().numpy()

    labels = ['alireza', 'amir', 'benyamin', 'hossein', 'maryam', 'mohammad', 'morteza', 'nahid', 'parisa', 'zahra', 'zeynab']
    output = np.argmax(preds)

print(labels[output])