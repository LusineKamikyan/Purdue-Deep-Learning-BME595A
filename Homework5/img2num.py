import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import time

class img2num(nn.Module):
    def __init__(self):
        super(img2num, self).__init__()
        # defining the CNN
        # add padding in the first layer to have 28x28 output instead of 24x24
        self.__layer1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5,stride = 1, padding = 2),
                nn.MaxPool2d(kernel_size=2, stride =2),
                nn.Tanh(),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.MaxPool2d(kernel_size=2, stride =2),
                nn.Tanh())

        self.__layer3 = nn.Sequential(
                nn.Linear(16*5*5,120),
                nn.Tanh(),
                nn.Linear(120,84),
                nn.Tanh(),
                nn.Linear(84,10)
                )

         
    def __forward_train(self, data):
        # move the image through the layers of the image        
        data = self.__layer1(data)
        # make the data into 1d for FC layers
        data = data.view(-1,16*5*5) 
        data = self.__layer3(data)
        return data       
        
        
    def train(self):
        BATCH_SIZE = 500
        model = self
        # load the data and normalize    
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    
        testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

        # define epochs, loss matrices, time matrix
        epoch = 10
        L = np.zeros((epoch,1))
        TL = np.zeros((epoch,1))
        Epoch = np.zeros((epoch,1))
        Run_time = np.zeros((epoch,1))
        
        # 28x28 -> 24x24 -> 14x14 -> 10x10 -> 5x5
        MSE_loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        #CE_loss = torch.nn.CrossEntropyLoss()
        
        for i in range(epoch): # epoch is 1000
            print(i)
            Loss = 0
            start_time = time.time()
            for batch_index, (data, target) in enumerate(trainloader):
                #zero the gradient so it doesn't accumulate
                optimizer.zero_grad()
                
                #make labels into onehot 
                labels_onehot = torch.FloatTensor(BATCH_SIZE, 10)
                target = target.view(1,BATCH_SIZE)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, torch.t(target), 1)
                # forward pass                
                output_forward = self.__forward_train(data)
                # calculate the MSE loss for training
                loss = MSE_loss(output_forward, labels_onehot)
                # add the losses of the batches
                Loss=Loss+loss
                # backward pass
                loss.backward()
                #update the parmameters
                optimizer.step()
            # average the losses
            L[i] = Loss.detach().numpy()/120
            # restart the loss for each epoch
            Loss = 0 
            Epoch[i]=i
            
            for batch_index, (data, target) in enumerate(testloader):
                #make labels into onehot
                labels_onehot = torch.FloatTensor(BATCH_SIZE, 10)
                target = target.view(1,BATCH_SIZE)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, torch.t(target), 1)
                #forward pass
                output_forward = self.__forward_train(data)
                # calculate the MSE loss for testing
                loss = MSE_loss(output_forward, labels_onehot)
                # add the losses
                Loss=Loss+loss
            end_time = time.time()
            # average over testing losses
            TL[i] = Loss.detach().numpy()/120
            Epoch[i]=i
            Run_time[i] = end_time-start_time
        
        #plot
        plt.figure(0)
        plt.plot(Epoch,L)
        plt.plot(Epoch,TL)
        plt.figure(1)
        plt.plot(Epoch,Run_time)
        
    def forward(self, img):
        # make it a float tensor
        img = img.type(torch.FloatTensor)
        # add extra 2 dimensions 
        img = img.view(1,1,28,28)
        # calles the forward train private function to run the forward pass
        output = self.__forward_train(img)
        return int(torch.argmax(output))

        
    
     
        
    
        
        
        
    