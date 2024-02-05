import torch.nn
import torch.nn.functional as F
x_data = torch.Tensor([[1], [.2], [.3], [0]])
y_data = torch.Tensor([[1], [.12], [.33], [.1]])
from tqdm import tqdm
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self, x):
        # y_pred = F.tanh(self.linear(x))
        # y_pred = F.relu(self.linear(x))
        # y_pred = F.relu6(self.linear(x))
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = Model()
# criterion = torch.nn.BCELoss(reduction='mean')
criterion = torch.nn.MSELoss(reduction='mean')
# criterion = torch.nn.BCELoss(reduction='mean')
# criterion = torch.nn.BCELoss(reduction='mean')
# criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr= .001)
# optimizer = torch.optim.Adam(model.parameters(), lr= .001)
# optimizer = torch.optim.Adamax(model.parameters(), lr= .001)
# optimizer = torch.optim.LBFGS(model.parameters(), lr= .001)

for epoch in tqdm(range(100000)):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.data)
    optimizer.zero_grad()
    optimizer.step()

print(loss.data)