from conv import Conv2D
import torch 
import cv2
import matplotlib.pyplot as plt
import scipy.misc
import time
import numpy as np


img = cv2.imread('pic1.jpg') #change to pic2 to run for the second picture
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
#plt.imshow(img)

img = img.astype(float)
tensor_img = torch.from_numpy(img)

tensor_img = tensor_img.type(torch.FloatTensor)


##### PART A ##### uncomment if you want to run, comment partB and partC
conv2d = Conv2D(3,1,3,1,'known')
[num_of_operation, output_image] = conv2d.forward(tensor_img)

for i in range(0,output_image.size(2)):
    result = output_image[:,:,i].numpy()
    scipy.misc.imsave("Pic1_Task_1_"+str(i+1)+".jpg", result)
    


##### PART B ##### uncomment if you want to run, comment partA and partC
'''TotalTime = np.zeros(shape = (11,1))
I = np.zeros(shape = (11,1))
for i in range(0,10):
    start_time = time.time()
    conv2d = Conv2D(3,2**i,3,1,'known')
    [num_of_operation, output_image] = conv2d.forward(tensor_img)
    end_time = time.time()
    TotalTime[i] = end_time-start_time
    print(TotalTime[i])
    #print(TotalTime)
    I[i] = i
    
plt.plot(I,TotalTime, 'o')'''




##### PART C ##### uncomment if you want to run, comment partA and partB
'''I = np.zeros(shape = (5,1))
N_OPER = np.zeros(shape=(5,1))
for i in range(2,5):
    conv2d = Conv2D(3,2,2*i+3,1,'rand')
    [num_of_operation, output_image] = conv2d.forward(tensor_img)
    N_OPER[i]= num_of_operation
    print(N_OPER[i])
    I[i] = 2*i+3
    
#print(N_OPER)
plt.plot(I, N_OPER, '-o')'''
    