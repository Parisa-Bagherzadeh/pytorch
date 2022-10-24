from statistics import variance
from turtle import forward
import torch
from torch.autograd import Variable

class LR (torch.nn.Module):
    def __init__(self) :
        super().__init__()
        self.linear = torch.nn.Linear(1,1)


    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred




if __name__ == '__main__':

    x = Variable(torch.Tensor([[1.0] , [2.0] , [3.0]]))
    y = Variable(torch.Tensor([[2.0] , [4.0] , [6.0]]))

    epochs = 200

    model = LR()

    loss_func = torch.nn.MSELoss(size_average = False)
    optimizer = torch.optim.SGD(model.parameters() , lr = 0.001)


    for epoch in range(epochs):
        
        pred_y = model(x)
        loss = loss_func(pred_y , y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1} , loss {loss.item()} ')



    new_var = Variable(torch.tensor([[10.0]]))    
    pred = model(new_var)
    print(pred.item())



