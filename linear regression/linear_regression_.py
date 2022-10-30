import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets


# 0 - prepare data
X_numpy , y_numpy = datasets.make_regression(n_samples = 100 , n_features = 1 , noise = 20 , random_state = 1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

#reshape y
y = y.view(y.shape[0],1)

n_sample , n_feature = X.shape


# 1 - model
input_size = n_feature
output_size = 1

model = nn.Linear(input_size,output_size)

# 2 - loss and optimizer
learning_rate = 0.01
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters() , lr = learning_rate)

# 3 - training loop
epochs = 100

for epoch in range(epochs):

    #forward pass and loss
    y_predicted = model(X)
    loss = loss_function(y_predicted,y)


    # backward pass
    loss.backward()


    # update 
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0 :
        print(f'epoch : {epoch + 1} , loss : {loss.item():.4f}')

#plot
predicted = model(X).detach().numpy()

fig = plt.figure(figsize = (8, 8))
plt.plot(X_numpy , y_numpy , 'ro')
plt.plot(X_numpy , predicted , 'b')
plt.show()

fig.savefig('linear_regression.png')


