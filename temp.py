import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# make the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(32*32*32, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 2)
        self.softmax = nn.LogSoftmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(-1, 32*32*32)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# create the instance of model
model = Net()

#generate the data
inp = torch.randn(128, 3, 32, 32)
targets = torch.LongTensor(128)
for i in xrange(0, inp.size()[0]):
    throw = np.random.uniform()
    if throw > 0.5:
        targets[i] = 1
    else:
        targets[i] = 0

train_list = torch.split(inp, 16, 0)
targets = torch.split(targets, 16)

# define an optimizer
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=3e-3, eps=1e-8, weight_decay=0.05)
avg_loss = list()

# train the model for some number of epochs
def train(epoch):
    for i, tr_batch in enumerate(train_list):
        data, t = Variable(tr_batch), Variable(targets[i])
        # do the forward pass
        scores = model.forward(data)
        loss = F.nll_loss(scores, t)
        # zero the grad parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.data[0])

        if i % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i) * len(data), len(inp),
                100. * (i)*16 / inp.size()[0], loss.data[0]))
            # plot the loss
            plt.plot(avg_loss)
            plt.savefig("avg_loss.jpg")
    plt.close()

if __name__ == '__main__':
    epoch = 100
    for i in xrange(epoch):
        train(i)
