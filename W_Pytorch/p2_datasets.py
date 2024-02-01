import torch
import torchvision
from torchvision import transforms, datasets 

transform = transforms.Compose([transforms.ToTensor()])

train = datasets.MNIST("", train=True, download = True, transform = transform)
test = datasets.MNIST("", train=False, download = True, transform = transform)

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)


for data in trainset:
    print(data)
    break
