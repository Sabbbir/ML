import torch
import torchvision
from torchvision import transforms, datasets 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



transform = transforms.Compose([transforms.ToTensor()])

train = datasets.MNIST("", train=True, download = True, transform = transform)
test = datasets.MNIST("", train=False, download = True, transform = transform)

trainset = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = True)
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
optimizer = optim.Adam(net.parameters(), lr = 0.001) # lr 0.001 is same as lr = 1e-3

epochs = 3
for epoch in range(epochs):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0
with torch.no_grad():
    X, y = data
    output = net(X.view(-1, 784))
    for idx, i in enumerate(output):
        if torch.argmax(i) == y[idx]:
            correct += 1
        total += 1
# print("Accuracy: ", round(correct/total), 3)
# print(X)
plt.imshow(X[0].view(28, 28))
plt.show()
print(torch.argmax(net(X[0].view(-1, 784))[0]))
# print(torch.argmax(net(X[0].view(-1, 784))[0]))