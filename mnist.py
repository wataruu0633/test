import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model_def import net
import numpy as np

train_data = MNIST('./mnist', train=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
test_data = MNIST('./mnist', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1)
device = torch.device("cuda")
net = net.to(device)
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 5000 == 4999:
            print('%d %d loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

        correct = 0
        total = 0
    for data in test_loader:
        inputs, labels = data
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy %d / %d = %f' % (correct, total, float(correct) / total))

First_Flg = True
for data in test_loader:
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    if First_Flg:
        OUT = outputs.cpu().data.numpy()
        LABEL = labels.cpu().data.numpy()
        PRED = predicted.cpu().data.numpy()
        First_Flg = False
    else:
        OUT = np.concatenate((OUT, outputs.cpu().data.numpy()), axis=0)
        LABEL = np.concatenate((LABEL, labels.cpu().data.numpy()), axis=0)
        PRED = np.concatenate((PRED, predicted.cpu().data.numpy()),axis=0)
for data in train_loader:
    inputs, labels = data
    inputs = Variable(inputs).to(device)
    labels = Variable(labels).to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    OUT = np.concatenate((OUT, outputs.cpu().data.numpy()), axis=0)
    LABEL = np.concatenate((LABEL, labels.cpu().data.numpy()), axis=0)
    PRED = np.concatenate((PRED, predicted.cpu().data.numpy()),axis=0)        
result = np.concatenate((OUT, LABEL.reshape(-1,1), PRED.reshape(-1,1)),axis=1)
np.save('result.npy', result)
