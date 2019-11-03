from NeuralNetwork import NeuralNetwork
import torch

class AND:
    
#def __init__(self):
# initialize the NeuralNetwork NN

#def __call__(self, x, y):
# call self.forward (x, y) and get the output of forward
# return the output of forward

#def forward(self, x, y):
# transfer (x, y) to 0, 1, and call NN.forward() do the computation of forward
# return the output of NN.forward()

#def train(self):
# compute the expected output of (x, y) by Python "and" operation
# call self.forward (x, y) and get the output of forward
# call NN.backward() to update dE_dTheta, delta
# call NN.updateParams() to update eta and perform Theta = Theta - eta * dE_dTheta
    
    
    def __init__(self):    
        ####initialize the NeuralNetwork NN
        self.__NN = NeuralNetwork([2,1])
        
    def __call__(self,x,y):
        ####call self.forward(x, y) and get the output of forward
        ####return T or F depending on the output of forward  
        output = self.forward(x,y)
        
        output_tensor = []
        for i in range(0, output.size(1)):
            if output[0,i] > 0.5:
                output_tensor.append(True)
            if output[0,i] < 0.5:
                output_tensor.append(False)
        return output_tensor
    
    def forward(self,x,y):
        ####transfer (x, y) to 0, 1, and call NN.forward() do the computation of forward
        ####return the output of NN.forward()
        
        input_tensor = torch.tensor([x,y])
        input_tensor = input_tensor.type(torch.FloatTensor)
        output = self.__NN.forward(input_tensor)
        #print(input_tensor)
        return output
    
    def train(self):
        # compute the expected output of (x, y) by Python "and" operation
       
        eta = 0.5
        x = [True,True,False,False,True,True,False,False,True,True,False,False]
        y = [True,False,True,False,True,False,True,False,True,False,True,False]
        
        target = torch.FloatTensor([True, False, False, False,True, False, False, False,True, False, False, False])
        # call self.forward (x, y) and get the output of forward
        for i in range(1000):
            output_forward = self.forward(x,y)
            # call NN.backward() to update dE_dTheta, delta
            update_dT_del = self.__NN.backward(target)
            # call NN.updateParams() to update eta and perform Theta = Theta - eta * dE_dTheta
            update_theta = self.__NN.updateParams(eta)
            
        print('updated theta')
        print(update_theta)
        

class OR:
    def __init__(self):
         # initialize the NeuralNetwork NN
        self.__NN = NeuralNetwork([2,1])
        
    def __call__(self,x,y):
        output = self.forward(x,y)
        
        #for i in range(0, len(output))
        output_tensor = []
        for i in range(0, output.size(1)):
            if output[0,i] > 0.5:
                output_tensor.append(True)
            if output[0,i] < 0.5:
                output_tensor.append(False)
        return output_tensor
        
    def forward(self,x,y):
        input_tensor = torch.tensor([x,y])
        input_tensor = input_tensor.type(torch.FloatTensor)
        output = self.__NN.forward(input_tensor)
 
        return output
    
    def train(self):
        # compute the expected output of (x, y) by Python "and" operation
       
        eta = 0.5
        x = [True,True,False,False,True,True,False,False,True,True,False,False]
        y = [True,False,True,False,True,False,True,False,True,False,True,False]
        
        target = torch.FloatTensor([True, True, True, False,True, True, True, False,True, True, True, False])
        # call self.forward (x, y) and get the output of forward
        for i in range(1000):
            output_forward = self.forward(x,y)
            # call NN.backward() to update dE_dTheta, delta
            update_dT_del = self.__NN.backward(target)
            # call NN.updateParams() to update eta and perform Theta = Theta - eta * dE_dTheta
            update_theta = self.__NN.updateParams(eta)
            
        print('updated theta')
        print(update_theta)
        
        
    
        
class XOR:
    def __init__(self):
         # initialize the NeuralNetwork NN
        self.__NN = NeuralNetwork([2,2,1])
        
    def __call__(self,x,y):
        output = self.forward(x,y)
        output_tensor = []
        for i in range(0, output.size(1)):
            if output[0,i] > 0.5:
                output_tensor.append(True)
            if output[0,i] < 0.5:
                output_tensor.append(False)
        return output_tensor
    def forward(self,x,y):
        input_tensor = torch.tensor([x,y])
        input_tensor = input_tensor.type(torch.FloatTensor)
        output = self.__NN.forward(input_tensor)
        
        return output
        
    def train(self):
        # compute the expected output of (x, y) by Python "and" operation
       
        eta = 0.2
        x = [True,True,False,False,True,True,False,False,True,True,False,False,True,True,False,False,True,True,False,False,True,True,False,False]
        y = [True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False]

        target = torch.FloatTensor([False, True, True, False, False, True, True, False,False, True, True, False,False, True, True, False,False, True, True, False,False, True, True, False])
        # call self.forward (x, y) and get the output of forward
        for i in range(5000):
            output_forward = self.forward(x,y)
            # call NN.backward() to update dE_dTheta, delta
            update_dT_del = self.__NN.backward(target)
            # call NN.updateParams() to update eta and perform Theta = Theta - eta * dE_dTheta
            update_theta = self.__NN.updateParams(eta)
            
        print('updated theta')
        print(update_theta)
 
        
        
    
        
class NOT:
    def __init__(self):
        self.__NN = NeuralNetwork([1,1])
        
           
    def __call__(self,x):
        output = self.forward(x)
        output_tensor = []
        for i in range(0, output.size(1)):
            if output[0,i] > 0.5:
                output_tensor.append(True)
            if output[0,i] < 0.5:
                output_tensor.append(False)
        return output_tensor
        
    def forward(self,x):
        input_tensor = torch.tensor([x])
        input_tensor = input_tensor.type(torch.FloatTensor)
        output = self.__NN.forward(input_tensor)
 
        return output
    
    def train(self):
        # compute the expected output of (x, y) by Python "and" operation
        eta = 0.5
        x = [True,False,True,False,True,False,True,False,True,False,True,False]
        #y = [False,True]
        
        target = torch.FloatTensor([False, True,False, True,False, True,False, True,False, True,False, True])
        # call self.forward (x, y) and get the output of forward
        for i in range(1000):
            output_forward = self.forward(x)
            # call NN.backward() to update dE_dTheta, delta
            update_dT_del = self.__NN.backward(target)
            # call NN.updateParams() to update eta and perform Theta = Theta - eta * dE_dTheta
            update_theta = self.__NN.updateParams(eta)
            
        print('updated theta')
        print(update_theta)

