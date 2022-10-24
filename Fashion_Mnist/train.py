from argparse import ArgumentParser

import torch
import torchvision

from model import MyModel



if __name__ =='__main__':

    batch_size = 64
    epochs = 20
    lr = 0.001

    parser = ArgumentParser()
    parser.add_argument("--device", type = str , default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)

    model = MyModel()
    model = model.to(device)
    model.train(True)


    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0),(1))

    ])

    dataset = torchvision.datasets.FashionMNIST('./dataset',train = True,download = True, transform = transform)
    train_data_loader = torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    loss_function = torch.nn.CrossEntropyLoss()


    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0

        for images,labels in train_data_loader:

            images = images.to(device)
            labels = labels.to(device) 
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_function(preds,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss
            train_acc += model.calc_acc(preds , labels)

        total_loss = train_loss / len(train_data_loader)
        total_acc = train_acc / len(train_data_loader)

        print(f'Epoch:{epoch+1} , Loss:{total_loss} , Accuracy:{total_acc}')

        torch.save(model.state_dict(),"weights.pth")




    







