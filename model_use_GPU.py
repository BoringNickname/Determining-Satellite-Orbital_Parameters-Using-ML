#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import pickle
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F 

print(torch.cuda.is_available())

lr = 0.0001
epochs = 100
batch_size = 10000

#IMPORTING DATA
#dataset variable should be a tuple of form (data, targets), each 1e5 long
with open('./datasets/list_orbital_dataset.pickle', 'rb') as data:
    dataset = pickle.load(data)

data, targets = dataset


#GET THE DEVICE (PREFEREABLY A GPU)
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

#CREATING A CUSTOM TRAINLOADER
class DatasetClass(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length

#CUSTOM TRAIN TEST SPLIT
def train_test_split_dataloaders(dataset, frac):
    train = torch.utils.data.Subset(dataset, range(0, int(frac*len(dataset))))
    test = torch.utils.data.Subset(dataset, range(int(frac*len(dataset)), len(dataset)))

    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    return trainloader, testloader

dataset = DatasetClass(data, targets)
trainloader, testloader = train_test_split_dataloaders(dataset,0.3)


#DEFINE A NEURAL NETWORK
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(50,10)
        self.fc2 = nn.Linear(10,6)
        # self.fc3 = nn.Linear(10,1)

    def forward(self,x):
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        return x

#A DIFFERENT ITERATION OF THE NETWORK
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # print('input shape', input_shape)
        self.cl1 = nn.Conv1d(1,3,3,1)

        self.cl2 = nn.Conv1d(3,5,3,1)
        self.fc1 = nn.Linear(20,6)

    def forward(self,x):
        x = x.view(batch_size, 1, 50)
        # print('batch input into the model', x.size())
        x = F.leaky_relu(self.cl1(x))
        x = torch.max_pool1d(x,3)
        x = F.leaky_relu(self.cl2(x))
        # print('after first conv1d', x.size())
        x = torch.max_pool1d(x,3)
        # print('after second conv1d', x.size())
        x = x.view(batch_size, -1)
        # print('size after flattening the conv layers', x.size())
        x = F.leaky_relu(self.fc1(x))
        return x


#INSTANTIATE THE MODEL
model = NeuralNetwork()
print('model name: ', model)
device = get_device()
print('device being used: ', device)
model.to(device)

#OPTIMIZER AND LOSS FUNCTION
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.MSELoss()
accuracy, train_losses, test_losses = [],[],[]

#%%

#TRAINING LOOP
for i in range(epochs):
    for j, (x_train, y_train) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(x_train.to(device))
        #calculate loss
        loss = loss_fn(output.to(device), y_train.to(device))

        #backpropagation
        loss.backward()
        optimizer.step()

    if i%10==0:
        train_losses.append(loss)
        print(f'epoch:{i}, loss:{loss}')        

print('Finished!')

#%%

for j, (x_test, y_test) in enumerate(testloader):
    output = model(x_test.to(device))

    print(output, y_test)
    break
