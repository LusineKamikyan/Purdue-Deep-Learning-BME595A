import torch

class NeuralNetwork:
    def __init__(self,size_list):
        self.__size_list = size_list
        
        #create an empty dictionary
        network = {}
        # puting random Thetas with mean 0 and std 1/(layer size)
        for i in range(len(self.__size_list)-1):
            layer_matrix = torch.normal(torch.zeros(self.__size_list[i+1],self.__size_list[i]+1),1/self.__size_list[i]**.5)
            layer_matrix = layer_matrix.type(torch.DoubleTensor)
            network["theta"+str(i+1)] = layer_matrix
        self.__network = network
    
    def getLayer(self,layer):
        theta_layer = self.__network["theta"+str(layer)]
        return theta_layer # weigth matrix from layer i to layer i+1
        
    def forward(self, input_tensor): # input_tensor is 1D or 2D tensor that are type double
        bias = torch.ones(1,input_tensor.size(1), dtype = torch.double)
        # attach input tensor and bias
        input_tensor =torch.cat((bias,input_tensor),0)
        
        for i in range(0,len(self.__size_list)-1):
            # get the theta
            theta_layer = self.getLayer(i+1)
            # theta * x
            y = torch.mm(theta_layer,input_tensor)
            #sigmoid
            output_tensor = torch.sigmoid(y) #1/(1+math.exp(-y))
            # input to the next layer plus bias
            input_tensor = torch.cat((bias,output_tensor),0)
            print(output_tensor)
        return output_tensor


    
    