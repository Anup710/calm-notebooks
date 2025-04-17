import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

print(X.shape, y.shape)
n_samples, n_features = X.shape

# test train split 

XT, Xt, yT, yt = train_test_split(X, y, test_size=0.2, random_state=37)
print(XT.shape, yT.shape, Xt.shape, yt.shape)

# scale features 
scaler = StandardScaler()
XT = scaler.fit_transform(XT)
Xt = scaler.transform(Xt)

# conversion into torch tensors and y into column vectors

XT = torch.from_numpy(XT.astype(np.float32))
Xt = torch.from_numpy(Xt.astype(np.float32))
yT = torch.from_numpy(yT.astype(np.float32))
yt = torch.from_numpy(yt.astype(np.float32))

yT = yT.view(yT.shape[0], 1)
yt = yt.view(yt.shape[0], 1)

# yT.shape, yt.shape
# XT.type(), yT.type()

## Custom log reg classifier model: 

class Model(nn.Module):
    def __init__(self, input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_features, 1)
    
    def forward(self, x):
        y_int = self.linear(x)
        y_pred = torch.sigmoid(y_int)
        return y_pred

# we have extracted the n_features above
log_reg_model = Model(n_features) #callable function -- will use in iteration loop with XT

# define loss and optimizer 
lr = 0.03

loss_fn = nn.BCELoss()
logreg_optimizer = torch.optim.SGD(log_reg_model.parameters(), lr = lr)

# define epochs and iteration loop: 

epochs = 100
i = 0

for i in range(epochs):
    
    # forward pass: calculate y_pred
    y_pred = log_reg_model(XT)
    #loss calculation
    bce_loss = loss_fn(y_pred, yT)
    # backprop
    bce_loss.backward()
    # weight, bias updateb
    logreg_optimizer.step()
    # gradient flush
    logreg_optimizer.zero_grad()

    if (i+1) % 10 == 0:
        print(f'epoch: {i+1}, loss = {bce_loss.item():.4f}')

with torch.no_grad():
    y_predicted = log_reg_model(Xt)
    y_pred_cls = y_predicted.round()
    t = (y_pred_cls == yt)
    accuracy = (t.sum()/ y_predicted.shape[0]) * 100
    print(f'Accuracy of log reg classifier = {accuracy.item():.4f}') 


