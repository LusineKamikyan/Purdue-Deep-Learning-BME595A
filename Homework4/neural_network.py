import torch
import numpy as np

class NeuralNetwork:
    def __init__(self,size_list):
        self.__size_list = size_list
        a = {} # activation
        a_hat = {}
        dE_dTheta = {}
        Theta = {}
        self.__Theta = Theta
        self.__a = a
        self.__dE_dTheta = dE_dTheta
        self.__a_hat = a_hat
        
        # puting random Thetas with mean 0 and std 1/(layer size)
        for i in range(len(self.__size_list)-1):
            layer_matrix = torch.normal(torch.zeros(self.__size_list[i+1],self.__size_list[i]+1),1/self.__size_list[i]**.5)
            #layer_matrix = torch.rand(self.__size_list[i+1],self.__size_list[i]+1)
            layer_matrix = layer_matrix.type(torch.FloatTensor)
            self.__Theta["theta"+str(i+1)] = layer_matrix
        
        for i in range(len(self.__size_list)-1):
            dtheta =  torch.zeros(self.__size_list[i+1],self.__size_list[i]+1)
            self.__dE_dTheta["layer"+str(i+1)] = dtheta
        
        
    def getLayer(self,layer):
        theta_layer = self.__Theta["theta"+str(layer)]
        
        return theta_layer # weigth matrix from layer i to layer i+1
        
    def forward(self, input_tensor): # input_tensor is 1D or 2D tensor that are type double
        bias = torch.ones(1,input_tensor.size(1), dtype = torch.float)
        # attach input tensor and bias
        input_tensor =torch.cat((bias,input_tensor),0)
        self.__a["layer"+str(1)] = input_tensor
        self.__a_hat["layer"+str(1)] = input_tensor
        for i in range(0,len(self.__size_list)-1):
            # get the theta
            theta_layer = self.getLayer(i+1)
            # theta * x
            #z_l
            y = torch.mm(theta_layer,input_tensor)
            #sigmoid
            output_tensor = torch.sigmoid(y) #1/(1+math.exp(-y))
            # input to the next layer plus bias
            # creating a_hat
            #input_tensor = torch.cat((bias,output_tensor),0)
            # activation of layer l+1 = i+2
            #output = torch.sigmoid(zz) #1/(1+math.exp(-z))
            self.__a["layer"+str(i+2)] = output_tensor
            #print(self.__a["layer"+str(i+2)])
            # a_hat of layer l+1 = i+2 is bias on top of a
            self.__a_hat["layer"+str(i+2)] = torch.cat((bias,output_tensor),0)
            # input to the next layer plus bias
            input_tensor = torch.cat((bias,output_tensor),0)
        return output_tensor
    
    def backward(self, target):
        # back-propagation pass single target (computes ∂E/∂Θ)
        delta = (self.__a["layer"+str(len(self.__size_list))] - target)*(self.__a["layer"+str(len(self.__size_list))]*(1-self.__a["layer"+str(len(self.__size_list))]))
       

        for i in range(len(self.__size_list)-1,0,-1):
            self.__dE_dTheta["layer"+str(i)] = torch.mm(delta,torch.t(self.__a_hat["layer"+str(i)]))
            if i >1:
                delta = torch.mm(torch.t(self.__Theta["theta"+str(i)][:,1:]),delta)*(self.__a["layer"+str(i)]*(1-self.__a["layer"+str(i)]))
            

    def updateParams(self,eta):
        for i in range(len(self.__size_list)-1,0,-1):
            self.__Theta["theta"+str(i)] = self.__Theta["theta"+str(i)]-eta/500*self.__dE_dTheta["layer"+str(i)]
        return self.__Theta