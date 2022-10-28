import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from utils import NNRegressor
from torch.autograd import Variable

class DnCNN(NNRegressor):

    def __init__(self, K, F):
        super(DnCNN, self).__init__()
        self.K = K
        self.F = F
        self.norm_net=0
        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, F, 3, padding=1))
        self.conv.extend([nn.Conv2d(F, F, 3, padding=1) for _ in range(K)])
        self.conv.append(nn.Conv2d(F, 3, 3, padding=1))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(F, F) for _ in range(K)])
        # initialize the weights of the Batch normalization layers
        for i in range(K):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(F))

    def forward(self, x):
        K = self.K
        h = functional.relu(self.conv[0](x))
        for i in range(K):
            h = functional.relu(self.bn[i](self.conv[i+1](h)))
        y = self.conv[K+1](h) + x
        return y

class unfolded_ISTA(NNRegressor):

    def __init__(self, K, F):
        super(unfolded_ISTA, self).__init__()
        self.K = K
        self.F = F
        self.norm_net=0
        # convolution layers
        self.conv = nn.ModuleList()
        for i in range(self.K):
            self.conv.append(nn.Conv2d(3,F,3,padding=1,bias=False))
            self.conv.append(nn.Conv2d(F, 3, 3, padding=1,bias=False))
            

        # self.conv.append(nn.Conv2d(C, 3, 3, padding=1,bias=False))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

    def forward(self, x):
        K = self.K
        # Initialization
        u = self.conv[0](x)
        #1st---> [K-1]-th layer
        for i in range(K):
            bk = self.conv[i*2](x)
            h = u-self.conv[i*2](self.conv[i*2+1](u))+ bk
            u = functional.hardtanh(h)
        # K-th layer
        y =  x-self.conv[2*K-1](u)
        return y
    
class unfolded_FISTA(NNRegressor):

    def __init__(self, K, F):
        super(unfolded_FISTA, self).__init__()
        self.K = K
        self.F = F
        self.norm_net=0
        # convolution layers
        self.conv = nn.ModuleList()
        for i in range(self.K):
            self.conv.append(nn.Conv2d(3,F,3,padding=1,bias=False))
            self.conv.append(nn.Conv2d(F, 3, 3, padding=1,bias=False))

        x=np.ones(K)
        self.multip = nn.Parameter(torch.tensor(x,requires_grad=True,dtype=torch.float))

        # apply He's initialization
        for i in range(K):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')


    def forward(self, x):
        K = self.K
        # Initialization
        u_prev = self.conv[0](x)
        y_prev = u_prev
        #1st---> [K-1]-th layer
        for i in range(K):
            temp = y_prev-self.conv[i*2](self.conv[i*2+1](y_prev))+ self.conv[i*2](x)
            u_next = functional.hardtanh(temp)
            y_next = (torch.ones_like(u_next)+self.multip[i].expand_as(u_next))*u_next-self.multip[i].expand_as(u_next)*u_prev
            u_prev = u_next
            y_prev = y_next
        # K-th layer
        y =  x-self.conv[2*K-1](u_prev)
        return y

class unfolded_CPv0(NNRegressor):

    def __init__(self, K, F):
        super(unfolded_CP, self).__init__()
        self.K = K
        self.F = F
        self.norm_net=0
        # convolution layers
        self.conv = nn.ModuleList()
        for i in range(self.K):
            self.conv.append(nn.Conv2d(3,F,3,padding=1,bias=False))
            self.conv.append(nn.Conv2d(F, F, 3, padding=1,bias=False))

        clone=np.ones(K)
        self.alpha = nn.Parameter(torch.tensor(clone,requires_grad=True,dtype=torch.float))
        self.sigma = nn.Parameter(torch.tensor(clone,requires_grad=True,dtype=torch.float))

        # apply He's initialization
        for i in range(K):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')


    def forward(self, x):
        K = self.K
        # Initialization
        x0 = x
        x1 = x
        u1 = self.conv[0](x)
        #1st---> [K-1]-th layer
        for i in range(K):
            temp = (torch.ones_like(x1)+self.alpha[i].expand_as(x1))*x1-self.alpha[i].expand_as(x0)*x0
            u2 = functional.hardtanh(u1+self.conv[i*2](temp))
            x2 = (self.sigma[i]/(self.sigma[i]+1))*x+(1/(1+self.sigma[i]))*x1- (self.sigma[i]/(self.sigma[i]+1))* self.conv[i*2+1](u2)

            x0= x1
            u1 =u2
            x1= x2

        # K-th layer
        return x1

class unfolded_ScCP(NNRegressor):

    def __init__(self, K, F):
        super(unfolded_ScCP, self).__init__()
        self.K = K
        self.F = F
        self.norm_net=0
        # convolution layers
        self.conv = nn.ModuleList()
        for i in range(self.K):
            self.conv.append(nn.Conv2d(3,F,3,padding=1,bias=False))
            self.conv.append(nn.Conv2d(F, 3, 3, padding=1,bias=False))

        x=np.ones(K)
        gamma = 1
        self.sigma = nn.Parameter(torch.tensor(x,requires_grad=True,dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(x,requires_grad=True,dtype=torch.float))

        # apply He's initialization
        for i in range(K):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')


    def forward(self, x):
        K = self.K
        # Initialization
        x0 = x
        x1 = x
        u1 = self.conv[0](x)
        #1st---> [K-1]-th layer
        gamma =1
        for i in range(K):
            if i>=1:
                self.alpha.data[i-1] = 1/torch.sqrt(1+2*gamma*self.sigma.data[i-1])
                self.sigma.data[i] = self.alpha.data[i-1]*self.sigma.data[i-1]
            temp = (torch.ones_like(x1)+self.alpha[i].expand_as(x1))*x1-self.alpha[i].expand_as(x0)*x0
            u2 = functional.hardtanh(u1+self.conv[i*2](temp))
            x2 = (self.sigma[i]/(self.sigma[i]+1)).expand_as(x)*x+(1/(1+self.sigma[i])).expand_as(x1)*x1- (self.sigma[i]/(self.sigma[i]+1)).expand_as(x)* self.conv[i*2+1](u2)

            x0= x1
            u1 =u2
            x1= x2

        # K-th layer
        return x1

class unfolded_CP(NNRegressor):

    def __init__(self, K, F):
        super(unfolded_CP, self).__init__()
        self.K = K
        self.F = F
        self.norm_net=0
        # convolution layers
        self.conv = nn.ModuleList()
        for i in range(self.K):
            self.conv.append(nn.Conv2d(3,F,3,padding=1,bias=False))
            self.conv.append(nn.Conv2d(F, 3, 3, padding=1,bias=False))

        x=np.ones(K)
        gamma = 1
        self.sigma = nn.Parameter(torch.tensor(x,requires_grad=True,dtype=torch.float))
        # apply He's initialization
        for i in range(K):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')


    def forward(self, x):
        K = self.K
        # Initialization
        x0 = x
        x1 = x
        u1 = self.conv[0](x)
        #1st---> [K-1]-th layer
        gamma =1
        for i in range(K):
            temp = x1-x0
            u2 = functional.hardtanh(u1+self.conv[i*2](temp))
            x2 = (self.sigma[i]/(self.sigma[i]+1)).expand_as(x)*x+(1/(1+self.sigma[i])).expand_as(x1)*x1- (self.sigma[i]/(self.sigma[i]+1)).expand_as(x)* self.conv[i*2+1](u2)

            x0= x1
            u1 =u2
            x1= x2

        # K-th layer
        return x1
