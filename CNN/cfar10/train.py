from model import ConvNet

from argparse import ArgumentParser

import torch
import torchvision


if __name__ == '__main__':

    batch_size = 4
    epochs = 50
    learning_rate = 0.001

    parser = ArgumentParser()
    parser.add_argument('--device', type = str , default = 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)

    model = ConvNet()
    model = model.to(device)
    model.train(True)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    dataset = torchvision.datasets.CIFAR10('./dataset' , train = True , download = True , transform = transform)
    train_data_loader = torch.utils.data.DataLoader(dataset , batch_size = batch_size , shuffle = True)
    optimizer = torch.optim.SGD(model.parameters() , lr = learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range (epochs):

        train_loss = 0.0
        train_acc = 0.0

        for images , labels in train_data_loader:
            images = images.to(device)
            lables = lables.to(device)

            preds = torch.model(images)
            optimizer.zero_grad()
            loss = loss_function(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_acc = model.acc(preds, labels)

        total_loss = train_loss / len(train_data_loader)
        total_acc = train_acc / len(train_data_loader)


        print(f'Epoch : {epoch+1} , Loss : {total_loss},Accuracy : {total_acc}')

    torch.save(model.state_dict(),'cnn_weights.pth')
        




