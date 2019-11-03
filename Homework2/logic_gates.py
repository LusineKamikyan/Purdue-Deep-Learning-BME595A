from neural_network import NeuralNetwork
import torch

class AND:
    def __init__(self):    
        # initialize the NeuralNetwork NN
        self.__NN = NeuralNetwork([2,1])
        #set manual weight
        theta_layer = self.__NN.getLayer(1)
        theta_layer[0,0] = -1
        theta_layer[0,1] = 0.6
        theta_layer[0,2] = 0.6
        
    def __call__(self,x,y):
        # call self.forward(x, y) and get the output of forward
        # return T or F depending on the output of forward  
        output = self.forward(x,y)
        if output > 0.5:
            output = True
        if output <0.5:
            output = False
        return output
    def forward(self,x,y):
        # transfer (x, y) to 0, 1, and call NN.forward() do the computation of forward
        # return the output of NN.forward()
        input_tensor = torch.tensor([[x],[y]])
        input_tensor = input_tensor.type(torch.DoubleTensor)
        output = self.__NN.forward(input_tensor)
 
        return output
        

class OR:
    def __init__(self):
         # initialize the NeuralNetwork NN
        self.__NN = NeuralNetwork([2,1])
        theta_layer = self.__NN.getLayer(1)
        theta_layer[0,0] = -1
        theta_layer[0,1] = 1.1
        theta_layer[0,2] = 1.1   
    def __call__(self,x,y):
        output = self.forward(x,y)
        if output > 0.5:
            output = True
        if output < 0.5:
            output = False
        return output
        
    def forward(self,x,y):
        input_tensor = torch.tensor([[x],[y]])
        input_tensor = input_tensor.type(torch.DoubleTensor)
        output = self.__NN.forward(input_tensor)
 
        return output
        
    
        
class XOR:
    def __init__(self):
         # initialize the NeuralNetwork NN
        self.__NN = NeuralNetwork([2,2,1])
        theta_layer1 = self.__NN.getLayer(1)
        theta_layer1[0,0] = -1                                                                                                               
        theta_layer1[0,1] = 1.1
        theta_layer1[0,2] = 1.1
        theta_layer1[1,0] = 1
        theta_layer1[1,1] = -0.7
        theta_layer1[1,2] = -0.7
        theta_layer2 = self.__NN.getLayer(2)
        theta_layer2[0,0] = -1
        theta_layer2[0,1] = 0.73
        theta_layer2[0,2] = 1.08
    def __call__(self,x,y):
        output = self.forward(x,y)
        if output > 0.5:
            output = True
        if output < 0.5:
            output = False
        return output
        
    def forward(self,x,y):
        input_tensor = torch.tensor([[x],[y]])
        input_tensor = input_tensor.type(torch.DoubleTensor)
        output = self.__NN.forward(input_tensor)
 
        return output
        
    
        
class NOT:
    def __init__(self):
        self.__NN = NeuralNetwork([1,1])
        theta_layer = self.__NN.getLayer(1)
        theta_layer[0,0] = 0.5
        theta_layer[0,1] = -1
           
    def __call__(self,x):
        output = self.forward(x)
        if output > 0.5:
            output = True
        if output < 0.5:
            output = False
        return output
        
    def forward(self,x):
        input_tensor = torch.tensor([[x]])
        input_tensor = input_tensor.type(torch.DoubleTensor)
        output = self.__NN.forward(input_tensor)
 
        return output