import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
#import time

class NnImg2Num: 
    def __init__(self):
        self.__BATCH_SIZE = 500
        self.__in_layer = 28*28
        self.__out_layer = 10
        self.__h1 = 100
        self.__model = nn.Sequential(
                     nn.Linear(self.__in_layer, self.__h1),
                     nn.Sigmoid(),
                     nn.Linear(self.__h1, self.__out_layer),
                     nn.Sigmoid(),
                     )

    def train(self):
        # load the data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.__BATCH_SIZE, shuffle=True)
    
        testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.__BATCH_SIZE, shuffle=False)

        epoch = 10
        L = np.zeros((epoch,1))
        TL = np.zeros((epoch,1))
        Epoch = np.zeros((epoch,1))
        #Run_time = np.zeros((epoch,1))
                     
        MSE_loss = torch.nn.MSELoss()
        optim = torch.optim.SGD(self.__model.parameters(), lr=0.5)
                
        for i in range(epoch): # epoch is 1000
            print(i)
            Loss = 0
            #start_time = time.time()
            for batch_index, (data, target) in enumerate(trainloader):
                #reshape the data
                data = data.view(self.__BATCH_SIZE, 784)
                
                #make labels into onehot
                labels_onehot = torch.FloatTensor(self.__BATCH_SIZE, 10)
                target = target.view(1,self.__BATCH_SIZE)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, torch.t(target), 1)
                labels_onehot = torch.t(labels_onehot)
                
                output_forward = self.__model(data)
                loss = MSE_loss(torch.t(output_forward), labels_onehot)
                Loss=Loss+loss
                #zero the gradient so it doesn't accumulate
                optim.zero_grad()
                # backward
                loss.backward()
                #update the parmameters
                optim.step()
            L[i] = Loss.detach().numpy()/120
            Loss = 0 
            for batch_index, (data, target) in enumerate(testloader):
            
                data = data.view(self.__BATCH_SIZE, 784)
                
                #make labels into onehot
                labels_onehot = torch.FloatTensor(self.__BATCH_SIZE, 10)
                target = target.view(1,self.__BATCH_SIZE)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, torch.t(target), 1)
                labels_onehot = torch.t(labels_onehot)
                
                output_forward = self.__model(data)
                loss = MSE_loss(torch.t(output_forward), labels_onehot)
                Loss=Loss+loss
            #end_time = time.time()

            TL[i] = Loss.detach().numpy()/120
            Epoch[i]=i
            #Run_time[i] = end_time-start_time
            
        plt.figure(0)
        plt.plot(Epoch,L)
        plt.plot(Epoch,TL)
        
        #plt.figure(1)
        #plt.plot(Epoch, Run_time)
        
        
    def forward(self,img):
        img = img.type(torch.FloatTensor)
        img = img.view(1,28*28)
        output = self.__model(img)
        
        return int(torch.argmax(output))
        
        