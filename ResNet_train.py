import os
import torch
torch.backends.cudnn.benchmark=True # accelerate
# import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import models as m
from torch.utils.data import DataLoader
# import auxiliary
import matplotlib.pyplot as plt # visualization
from time import time
import datetime
import random
import numpy as np
import pandas as pd
import gc # garbage collector
from torchinfo import summary

# Global random number seeds provide limited control
# Does not fully stabilise the model
torch.manual_seed(1412) #torch
random.seed(1412) #random
np.random.seed(1412) #numpy.random

torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping():
    def __init__(self, patience=5, tol=0.0005):

        # When the difference between this iteration's loss and the minimum loss is less than a threshold
        # an early stop is set off.

        self.patience = patience
        self.tol = tol  # tolerance
        self.counter = 0
        self.lowest_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.lowest_loss == None:
            self.lowest_loss = val_loss
        elif self.lowest_loss - val_loss > self.tol:
            self.lowest_loss = val_loss
            self.counter = 0
        elif self.lowest_loss - val_loss < self.tol:
            self.counter += 1
            print("\t NOTICE: Early stopping counter {} of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                print('\t NOTICE: Early Stopping Actived')
                self.early_stop = True
        return self.early_stop


# import data
train = torchvision.datasets.MNIST(root= r'./datasets'
                                   , train=True
                                   , transform=T.ToTensor()
                                   , download=False
                                   )
test = torchvision.datasets.MNIST(root= r'./datasets'
                                  , train=False
                                  , transform=T.ToTensor()
                                  , download=False
                                  )
train_loader = DataLoader(dataset=train,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(dataset=test,
                         batch_size=64,
                         shuffle=True)

# construct network
class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)                       # [200, 64, 28, 28]
        x = self.block2(x)                       # [200, 128, 14, 14]
        x = self.block3(x)                       # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)                   # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)                # [200,256]
        out = self.outlayer(x)
        return out

ResNet18 = ResNet(Basicblock, [1, 1, 1, 1], 10)

summary(ResNet18,(10,1,32,32),depth=2,device="cpu")
# print(ResNet18)

def plotloss(trainloss, testloss):
    plt.figure(figsize=(10, 7))
    plt.plot(trainloss, color="red", label="Trainloss")
    plt.plot(testloss, color="orange", label="Testloss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

torch.cuda.manual_seed(1412)
torch.cuda.manual_seed_all(1412)
torch.cuda.is_available()


def IterOnce(net, criterion, opt, x, y):
    """
    train phase

    net: model
    criterion: loss function
    opt: optimization
    x: all samples in one batch
    y: all labels in one batch
    """
    sigma = net.forward(x)
    loss = criterion(sigma, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)  # save the memory
    yhat = torch.max(sigma, 1)[1]
    correct = torch.sum(yhat == y)
    return correct, loss


def TestOnce(net, criterion, x, y):
    """
    Test Phase

    net: model
    criterion: loss function
    x: all samples in one batch
    y: all labels in one batch
    """

    with torch.no_grad(): # accelerate
        sigma = net.forward(x)
        loss = criterion(sigma, y)
        yhat = torch.max(sigma, 1)[1]
        correct = torch.sum(yhat == y)
    return correct, loss

def fit_test(net, batchdata, testdata, criterion, opt, epochs, tol, modelname, PATH):
    """
    Train the model and output the accuracy/loss on the training and test sets after each epoch
    Enable monitoring of the model
    Save model

    Parameters：
    net: Neural Network
    batchdata：train set
    testdata：test set
    criterion：loss function
    opt：optimization
    epochs：number of epochs
    tol：early stop
    modelname：save the parameters of the trained model
    PATH：the directory of the trained model

    """

    SamplePerEpoch = batchdata.dataset.__len__()
    allsamples = SamplePerEpoch * epochs
    trainedsamples = 0
    trainlosslist = []
    testlosslist = []
    early_stopping = EarlyStopping(tol=tol)
    highestacc = None

    for epoch in range(1, epochs + 1):
        net.train()
        correct_train = 0
        loss_train = 0
        for batch_idx, (x, y) in enumerate(batchdata):
            # non_blocking = True
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(x.shape[0])
            correct, loss = IterOnce(net, criterion, opt, x, y)

            trainedsamples += x.shape[0]
            loss_train += loss
            correct_train += correct

            if (batch_idx + 1) % 125 == 0:
                print('Epoch{}:[{}/{}({:.0f}%)]'.format(epoch
                                                        , trainedsamples
                                                        , allsamples
                                                        , 100 * trainedsamples / allsamples))

        TrainAccThisEpoch = float(correct_train * 100) / SamplePerEpoch
        TrainLossThisEpoch = float(loss_train * 100) / SamplePerEpoch
        trainlosslist.append(TrainLossThisEpoch)


        # Clear out intermediate variables under an epoch loop that are no longer needed
        del x, y, correct, loss, correct_train, loss_train  # delete data and variables
        gc.collect()  # delete data and cache
        torch.cuda.empty_cache()  # Free the memory allocated


        net.eval()
        correct_test = 0
        loss_test = 0
        TestSample = testdata.dataset.__len__()

        for x, y in testdata:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(x.shape[0])
            correct, loss = TestOnce(net, criterion, x, y)
            loss_test += loss
            correct_test += correct

        TestAccThisEpoch = float(correct_test * 100) / TestSample
        TestLossThisEpoch = float(loss_test * 100) / TestSample
        testlosslist.append(TestLossThisEpoch)

        # clean
        del x, y, correct, loss, correct_test, loss_test
        gc.collect()
        torch.cuda.empty_cache()

        # print
        print("\t Train Loss:{:.6f}, Test Loss:{:.6f}, Train Acc:{:.3f}%, Test Acc:{:.3f}%".format(TrainLossThisEpoch
                                                                                                   , TestLossThisEpoch
                                                                                                   , TrainAccThisEpoch
                                                                                                   , TestAccThisEpoch))

        # Save
        if highestacc == None:
            highestacc = TestAccThisEpoch
        if highestacc < TestAccThisEpoch:
            highestacc = TestAccThisEpoch
            torch.save(net.state_dict(),os.path.join(PATH,modelname+".pt"))

            print("\t Weight Saved")


        early_stop = early_stopping(TestLossThisEpoch)
        if early_stop == "True":
            break

    print("Complete")
    return trainlosslist, testlosslist


def full_procedure(net, epochs, bs, modelname, PATH, lr=0.001, alpha=0.99, gamma=0, wd=0, tol=10 ** (-5)):
    torch.cuda.manual_seed(1412)
    torch.cuda.manual_seed_all(1412)
    torch.manual_seed(1412)

    # split
    batchdata = DataLoader(train, batch_size=bs, shuffle=True
                           , drop_last=False, pin_memory=True)
    testdata = DataLoader(test, batch_size=bs, shuffle=False
                          , drop_last=False, pin_memory=True)

    # loss & optimization
    criterion = nn.CrossEntropyLoss(reduction="sum")
    opt = optim.RMSprop(net.parameters(), lr=lr
                        , alpha=alpha, momentum=gamma, weight_decay=wd)

    # train & test
    trainloss, testloss = fit_test(net, batchdata, testdata, criterion, opt, epochs, tol, modelname, PATH)

    return trainloss, testloss

'''
===========TIME WARNING==========

MyResNet
 1 epoch on CPU: 9min50s~10mins
 1 epoch on GPU: 18s~20s

===========TIME WARNING===========
'''


modelname = "MyResNet_MNIST_clean"
PATH = r'.'
print(modelname)
torch.manual_seed(1412)

ResNet18 = ResNet(Basicblock, [1, 1, 1, 1], 10)
net2 = ResNet18.to(device,non_blocking=True)
start = time() # calculate time
trainloss, testloss = full_procedure(net2,epochs=30, bs=256
                                     ,modelname=modelname
                                     ,PATH = PATH
                                     ,tol = 10**(-10))
print(time()-start)
plotloss(trainloss,testloss)