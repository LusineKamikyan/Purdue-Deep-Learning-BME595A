import torchvision
import torchvision.transforms as transforms
import torch
from neural_network import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt
#import time

class MyImg2num: 
    def __init__(self):
        self.__NN = NeuralNetwork([28*28,100,10])
        self.__BATCH_SIZE = 500

    def train(self):
        #transform = transforms.ToTensor()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.__BATCH_SIZE, shuffle=True)
    
        testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.__BATCH_SIZE, shuffle=False)

        
        eta = 0.5
        epoch = 50
        L = np.zeros((epoch,1))
        TL = np.zeros((epoch,1))
        Epoch = np.zeros((epoch,1))
        #Run_time = np.zeros((epoch,1))
        
        for i in range(epoch):
            print(i)
            Loss = 0
            #start_time = time.time()
            for batch_index, (data, target) in enumerate(trainloader): #trainloader shuffles the data
            #get bacth images and their lables
                #reshape data
                data = data.view(self.__BATCH_SIZE, 784)
                
                #make labels into onehot
                labels_onehot = torch.FloatTensor(self.__BATCH_SIZE, 10)
                target = target.view(1,self.__BATCH_SIZE)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, torch.t(target), 1)
                labels_onehot = torch.t(labels_onehot)
                
                #forward, backward, update
                output_forward = self.__NN.forward(torch.t(data))
                # call NN.backward() to update dE_dTheta, delta
                self.__NN.backward(labels_onehot)
                # call NN.updateParams() to update eta and perform Theta = Theta - eta * dE_dTheta
                self.__NN.updateParams(eta)
                #output_forward = torch.t(output_forward)
                # mse error
                E_batch = (0.5*((output_forward-labels_onehot)**2)).mean()
        
                Loss = Loss+E_batch
                
            L[i] = Loss.mean()    
            Epoch[i] = i
            
            
            Loss = 0
            for batch_index, (data, target) in enumerate(testloader): 
                #reshape data
                data = data.view(self.__BATCH_SIZE, 784)
               
                #make labels into onehot
                labels_onehot = torch.FloatTensor(self.__BATCH_SIZE, 10)
                target = target.view(1,self.__BATCH_SIZE)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, torch.t(target), 1)
                labels_onehot = torch.t(labels_onehot)
                
                # going through the forward
                output_forward = self.__NN.forward(torch.t(data))
                #output_forward = torch.t(output_forward)
               
                E_batch = (0.5*((output_forward-labels_onehot)**2)).mean()
                    
                Loss = Loss+E_batch
            #end_time = time.time()
            #Run_time[i] = end_time-start_time
            TL[i] = Loss.mean()    

        plt.figure(0)
        plt.plot(Epoch,TL)    
        plt.plot(Epoch,L)
        #plt.figure(1)
        #plt.plot(Epoch,Run_time)
        
        return Epoch, L, TL
        
    def forward(self,img):
        img = img.type(torch.FloatTensor)
        img = img.view(28*28,1)
        output = self.__NN.forward(img)
        
        
        return int(torch.argmax(output))
        
