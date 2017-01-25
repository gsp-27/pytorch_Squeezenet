import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import model
import torch.nn.functional as F
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size of train')
parser.add_argument('--epoch', type=int, default=55, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.0003, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use cuda for training')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of epochs to save snapshot after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../', train=False, transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# get the model and convert it into cuda for if necessary
net  = model.SqueezeNet()
if args.cuda:
    net.cuda()
#print(net)

# create optimizer
optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.999), lr=3e-3, eps=1e-8, weight_decay=0.05)
avg_loss = list()

# train the model
# TODO: Compute training accuracy and test accuracy
# TODO: train it on some data and see if it overfits.
# TODO: train the data on final model

def train(epoch):
    global avg_loss
    net.train()
    for b_idx, (data, targets) in enumerate(train_loader):
        # convert the data and targets into Variable and cuda form
        data, targets = Variable(data), Variable(targets)
        if args.cuda:
            data.cuda(), targets.cuda()
        # train the network
        scores = net.forward(data)
        loss = F.nll_loss(scores, targets)
        avg_loss.append(loss.data[0])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1) / len(train_loader), loss.data[0]))
            # also plot the loss, it should go down exponentially at some point
            plt.plot(avg_loss)
            plt.save_fig("Squeezenet_loss")
    plt.close()

if __name__ == '__main__':
    for i in xrange(1):
        train(i)
