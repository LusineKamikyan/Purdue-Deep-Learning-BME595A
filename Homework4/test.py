import torch
import torchvision
import torchvision.transforms as transforms
from my_img2num import MyImg2num
from nn_img2num import NnImg2Num
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
from neural_network import NeuralNetwork as NN
 
#myimg2num = MyImg2num()
#myimg2num.train()


nnimg2num = NnImg2Num()
nnimg2num.train()


transform = transforms.ToTensor()

test_set = torchvision.datasets.MNIST('/tmp', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False)
dataiter = iter(testloader)
images, data = dataiter.next()

counter = 0
for i in range(10000):
    a = nnimg2num.forward(images[i,0,:,:])  
    if data[i] ==a:
        counter+=1


