import torch
import torchvision
from torchvision import transforms, datasets 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


transform = transforms.Compose([transforms.ToTensor()])

train = datasets.MNIST("", train=True, download = True, transform = transform)
test = datasets.MNIST("", train=False, download = True, transform = transform)

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # FC is fully connected
        # 764 is 28*28 input layer size
        # 64 is output/next layer size
        # self.fc1 = nn.Linear(input dimension, output layer dimension)
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        #Logics can be thrown here using torch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #for example
        # if x is not y:
            # x = F.relu(self.fc3(x))
            # x = self.fc4(x)
        # Possible
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x,  dim=1)



net = Net()
# print(net(X))
 
# X = torch.rand((28,28))
# X = X.view(1, 28*28)
# 
# output = net(X)
# print(output)

