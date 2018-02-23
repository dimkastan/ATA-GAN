"""

 Author: Dimitris Kastaniotis

If you use our method please consider to cite our work:

Dimitris Kastaniotis, Ioanna Ntinou, Dimitris Tsourounis, George Economou and Spiros Fotopoulos, 
Attention-Aware Generative Adversarial Nets (ATA-GANs), submitted to IVMSP 2018


"""
import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
from torch.optim import lr_scheduler
from models_with_cam import *
import numpy as np 
import numpy


#-----------------------------------------------
#                       fGenerator model
#-----------------------------------------------
class generatorV2(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generatorV2, self).__init__()
        self.upsample   = nn.Upsample(scale_factor=2,mode = "nearest")
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.Conv2d(d*8, d*8, 3, 1, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*8)
        self.deconv3 = nn.Conv2d(d*8, d*4, 3, 1, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*4)
        self.deconv4 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm2d(d*2)
        self.deconv5 = nn.Conv2d(d*2, d, 3, 1, 1)
        self.deconv5_bn = nn.BatchNorm2d(d)
        self.deconv6 = nn.Conv2d(d, 1, 3, 1, 1)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = self.upsample(x)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = self.upsample(x)
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.upsample(x)
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.upsample(x)
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.tanh(self.deconv6(x))
        return x

#-----------------------------------------------
#                       Disriminator model
#-----------------------------------------------
class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d*8, 3, 2, 1)
        self.conv2 = nn.Conv2d(d*8, d*8, 3, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*8)
        self.conv3 = nn.Conv2d(d*8, d*4, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d*2)
        self.conv5 = nn.Conv2d(d*2, d*2, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d*2)
        self.conv6 = nn.Conv2d(d*2, d*2, 3, 1, 1)
        self.avgpool = nn.AvgPool2d(16 , stride=1)
        self.fc = nn.Linear(d*2, 1)
        self.upsample   = nn.Upsample(scale_factor=4,mode = "bilinear")
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        # print("x ={}".format(x.shape))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        # print("x ={}".format(x.shape))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # print("x ={}".format(x.shape))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x =   self.conv6(x)
        #x = F.leaky_relu(self.conv7_bn(self.conv7(x)), 0.2)
        # print("before pool x ={}".format(x.shape))
        x1= self.avgpool(x)
        # print("After poolx ={}".format(x.shape))
        x1 = x1.view(x1.size(0), -1)
        outsm = F.sigmoid(self.fc(x1))
        w = torch.mm(outsm, Variable(self.fc.weight.data))
        cam = torch.mul(x,w.unsqueeze(2).unsqueeze(3))
        cam = cam.sum(1).unsqueeze(1)
        # print("OK")
        #print("outputCAM size is {}".format(self.upsample(cam).shape))
        return outsm, self.upsample(cam) 

 
 



transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(66),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
])
          
# first unzip the data into the same directory
data_dir = 'WithoutMasks/TrainVal/val'
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=25, shuffle=True,
    num_workers= 4,pin_memory=False)
print(train_loader.dataset.imgs[0][0])
temp = plt.imread(train_loader.dataset.imgs[0][0])
 
#-----------------------------------------------
#                           networks
#-----------------------------------------------
G = generatorV2(64)
D = discriminator(128)
 

rnet = resnet18(True, num_classes=6);

rnet.load_state_dict(torch.load("best_classifier.pt"));

 
D.load_state_dict(torch.load("best_discriminator.pt"))
G.load_state_dict(torch.load("best_generator.pt"))
 

 
#-----------------------------------------------
#                           move to cuda
#-----------------------------------------------
rnet.cuda()
G.cuda()
D.cuda()
G.train(False)
D.train(False)
rnet = rnet.cuda()
rnet.train(False)
 
 
 
#-----------------------------------------------
#                       function to draw pairs
#-----------------------------------------------
def plotpairs(images, cams, rows, cols):
    fig, ax = plt.subplots(rows*2, cols, figsize=(rows, cols))
    ccnt=0;
    for i, j in itertools.product(range(rows), range(cols)):
        ax[i*2, j].get_xaxis().set_visible(False)
        ax[i*2, j].get_yaxis().set_visible(False)
        ax[i*2, j].cla()
        ax[i*2+1, j].cla()
        ccnt=ccnt+1
        ax[i*2, j].imshow(images[ccnt][0])
        ax[i*2+1, j].imshow(cams[ccnt][0])

 
 
#-----------------------------------------------
#          Iterate over all data and store results into png images
#-----------------------------------------------
for x_, labels in train_loader:
    mini_batch = x_.size()[0]
    y_real_ = torch.ones(mini_batch)
    y_fake_ = torch.zeros(mini_batch)
    x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
    classpred, cams = rnet(x_)
    plt.figure(1)
    plotpairs(x_.data.cpu(), cams.data.cpu(), 5 , 5)
    plt.savefig('TEACHERnet_cams.png') 
    _, preds = torch.max(classpred.data, 1)
    labels = Variable(labels.cuda())
    D_result , camDReal= D(x_)
    plt.figure(2)
    plotpairs(x_.data.cpu(), camDReal.data.cpu(), 5 , 5)
    plt.savefig('REAL_D_cams.png') 
    plt.close()     
    z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda())
    G_result = G(z_)     
    D_result , CAMfake= D(G_result)
    plt.figure(3)
    plotpairs(x_.data.cpu(), CAMfake.data.cpu(), 5, 5)
    plt.savefig('FAKE_D_cams.png') 
    plt.close()




 