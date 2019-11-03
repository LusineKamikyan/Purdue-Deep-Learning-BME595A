import torch 
import numpy as np
import random as rand

class Conv2D:
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        # in_channel = depth of image, RGB
        # o_channel = number of kernels, the same as output image depth
        # kernel_size = width of kernel
        # stride = by how much we move the kernel vertically and horizontally
        # mode = random vs known kernel
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.k_size = kernel_size
        self.stride = stride
        self.mode = mode
        
        
    def forward(self, input_image):
        
        [row, col,hgt] = input_image.size()
           
        K = Conv2D.kern(self)
        K = K.type(torch.Tensor)
        
        output_image = torch.zeros((row-self.k_size)//self.stride+1,(col-self.k_size)//self.stride+1, self.o_channel)
        
        count = 0
        for j in range(0,col-self.k_size+1,self.stride): #looping through columns
            for i in range(0,row-self.k_size+1, self.stride): #looping through rows 
                for l in range(0,K.size(0)): #layers of the kernel
                    sum_in_depth = 0
                    for k in range(0,hgt): #layers of the picture
                        dot_pr = torch.sum(input_image[i:i+self.k_size,j:j+self.k_size,k]*K[l,:,:])
                        sum_in_depth+=dot_pr 
                        # counting the operations
                        count += 2*self.k_size**2
                    count = count-1  
                    
                    # saving the result in the corresponging entry of output image
                    output_image[i//self.stride,j//self.stride,l] = sum_in_depth
                
        print(output_image.size())
        return count, output_image
        #return [int, 3D FloatTensor]
     
    ## function kern() return the kernel(s) needed to be used
    def kern(self):
        if self.mode == 'known':
            if self.o_channel == 1:
                k1 = np.array([[-1., -1., -1.],[0., 0., 0.],[1., 1., 1.]])
                k1_tens = torch.from_numpy(k1)
                tensor_list = [k1_tens]
                kernel = torch.stack(tensor_list)
            elif self.o_channel == 2:
                k4 = np.array([[-1., -1., -1., -1., -1.],[-1., -1., -1., -1., -1.],[0., 0., 0., 0., 0.],[ 1., 1., 1., 1., 1.],[1., 1., 1., 1., 1.]])
                k5 = np.array([[-1., -1., 0., 1., 1.],[-1., -1., 0., 1., 1.],[-1., -1., 0., 1., 1.],[ -1., -1., 0., 1., 1.],[-1., -1., 0., 1., 1.]])
                # convert to tensor
                k4_tens = torch.from_numpy(k4)
                k5_tens = torch.from_numpy(k5)  
                # stack the tensors to get width of o_channel tensor
                tensor_list = [k4_tens, k5_tens]
                kernel = torch.stack(tensor_list)
            elif self.o_channel == 3:
                k1 = np.array([[-1., -1., -1.],[0., 0., 0.],[1., 1., 1.]])
                k2 = np.array([[-1.,  0.,  1.], [-1., 0., 1.], [-1., 0., 1.]])
                k3 = np.array([[ 1.,  1.,  1.],[1., 1., 1.], [1., 1., 1.]])
                k1_tens = torch.from_numpy(k1)
                k2_tens = torch.from_numpy(k2)
                k3_tens = torch.from_numpy(k3)
                tensor_list = [k1_tens, k2_tens, k3_tens]
                kernel = torch.stack(tensor_list)
        else:
            kernel = torch.zeros(self.o_channel,self.k_size,self.k_size, dtype = torch.float)
            for j in range(0,self.o_channel): 
                kernel[j,:,:] = torch.rand(self.k_size,self.k_size, dtype = torch.float)
                    
        return kernel