from NeuralNetwork import NeuralNetwork
import torch
from logicGates import AND
from logicGates import OR
from logicGates import XOR
from logicGates import NOT


##### PART A #####
model = NeuralNetwork([2,4,1])
input_tensor= torch.tensor([[0.05,2],[0.1,0.5]], dtype = torch.float)
output = model.forward(input_tensor)
target = torch.tensor([[0.01,0.03]], dtype = torch.float)
y=model.backward(target)
#print(model.updateParams(0.3))


##### PART B #####
print("Results for AND:")
And = AND()
And.train()
print(And([False,True,True, False,False,True],[True,True,True,False,False,False]))
print(And([True],[False]))
print(And([True],[True]))
print(And([False],[False]))
print(And([True],[False]))


print("Results for OR:")
Or = OR()
Or.train()
print(Or([False,True,True, False,False,True],[True,True,True,False,False,False]))
print(Or([True],[True]))
print(Or([False],[False]))
print(Or([True],[False]))
print(Or([False],[True]))


print("Results for NOT:")
Not = NOT()
Not.train()
print(Not([True,False,True,False,False,False]))
print(Not([True]))
print(Not([False]))


print("Results for XOR:")
Xor = XOR()
Xor.train()
print(Xor([True,True,False,False,False,True],[True,True,True,False,True,False]))
print(Xor([False],[False]))
print(Xor([True],[True]))
print(Xor([True],[False]))
print(Xor([False],[True]))