#import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import time


class img2obj(nn.Module):
    def __init__(self):
        # define the CNN
        super(img2obj, self).__init__()
        self.__layer1 = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5),
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
                nn.Linear(84,100))

         
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
        #load and normalize
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        trainset = torchvision.datasets.CIFAR100('/tmp', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    
        testset = torchvision.datasets.CIFAR100('/tmp', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

        epoch = 25
        L = np.zeros((epoch,1))
        TL = np.zeros((epoch,1))
        Epoch = np.zeros((epoch,1))
        Run_time = np.zeros((epoch,1))
        
        
        # 32x32 -> 28x28 -> 14x14 -> 10x10 -> 5x5
        # optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        CE_loss = torch.nn.CrossEntropyLoss()
        
        for i in range(epoch): # epoch is 1000
            print(i)
            Loss = 0
            start_time = time.time()
            for batch_index, (data, target) in enumerate(trainloader):
                # forward pass through CNN
                output_forward = self.__forward_train(data)
                # calculate the CE loss
                loss = CE_loss(output_forward, target)
                # add the losses uo to each other
                Loss=Loss+loss
                #zero the gradient so it doesn't accumulate
                optimizer.zero_grad()
                # backward pass
                loss.backward()
                #update the parmameters
                optimizer.step()
            # average the losses
            L[i] = Loss.detach().numpy()/120
            print(L[i])
            Loss = 0 
            Epoch[i]=i

            for batch_index, (data, target) in enumerate(testloader):
                # forward pass through CNN
                output_forward = self.__forward_train(data)
                # calculate the CE loss
                loss = CE_loss(output_forward, target)                
                # add looses to each other
                Loss=Loss+loss
            end_time = time.time()
            # average losses
            TL[i] = Loss.detach().numpy()/120
            print(TL[i])
            # time to run through test and train
            Run_time[i] = end_time-start_time
            
        plt.figure(0)
        plt.plot(Epoch,L)
        plt.plot(Epoch,TL)

        
        # [3x32x32 ByteTensor] img
        # returns a string
    def forward(self, img):
        # define the categories
        labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm']
        # make the byte tensor to float
        img = img.type(torch.FloatTensor)
        # change dimensions
        img = img.view(1,3,32,32)
        # go through the forward pass
        output = self.__forward_train(img)
        # return the string corresponding to the prediciton
        return labels[int(torch.argmax(output))]

#        
#    # img is 3x32x32 float tensor
    def view(self, img):
        prediction_image = self.forward(img)
        img = img.type(torch.FloatTensor)
        #convert to numpy
        img = img.numpy()
        # output the image with its label on a window
        cv2.namedWindow(prediction_image, cv2.WINDOW_NORMAL)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        cv2.imshow(prediction_image,img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
       
    def cam(self, index=0):
        # start the camera
        camera = cv2.VideoCapture(index)
        print("\nPress q to quit video capture\n")
        # camera window setting
        font = cv2.FONT_HERSHEY_SIMPLEX 
        camera.set(3, 1280)
        camera.set(4, 720)
        while True:
            # continuous frame capture
            retval, frame = camera.read()
            if retval:
                # resize the image to 32x32
                image = cv2.resize(frame, (32,32))
                # normalize, make into a float tensor, add extra dimension
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                image = transform(image)
                image = image.unsqueeze(0)
                # forward pass to predict the label
                predicted_image = self.forward(image)
                # show the image
                cv2.putText(frame, predicted_image,(250, 50),font, 2, (255, 200, 100), 5, cv2.LINE_AA)
                cv2.imshow('Image', frame) 
                
            else:
                raise ValueError("Can't read frame")
                break
            key_press = cv2.waitKey(1)
            if key_press == ord('q'):
                break
            
        camera.release()
        cv2.destroyAllWindows() # Closing the window
        
