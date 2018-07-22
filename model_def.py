from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5) #28*28 -> 24*24
        self.pool1 = nn.MaxPool2d(2, 2) #24*24 -> 12*12
        self.conv2 = nn.Conv2d(5, 5, 5) #12*12 -> 8*8
        self.pool2 = nn.MaxPool2d(2,2) #8*8 -> 4*4
        self.l1 = nn.Linear(5*4*4, 40)
        self.l2 = nn.Linear(40, 10)
        
    def forward(self, x):
#        x = x.view(-1, 28 * 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 5*4*4)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        softmax = nn.Softmax(dim=1)
        x = softmax(x)
        return x
    
net = Net()
