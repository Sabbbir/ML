import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available. Switching to CPU.")

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Increase the number of workers for data loading
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
# Create an instance of the neural network
net = Net()

# Check if GPU is available and move the model to GPU
if torch.cuda.is_available():
    print("Moving model to GPU.")
    net.cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in tqdm(range(50)):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        start_time = time.time()
        inputs, labels = data

        # Move input data and labels to GPU
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # Print the device of tensors to verify GPU usage
        # print("Input device:", inputs.device)
        # print("Labels device:", labels.device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # end_time = time.time()
        # print(f"Data loading time: {end_time - start_time} seconds")

        if i % 200 == 199:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            # start_time = time.time()

print("Finished Training")
