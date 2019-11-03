from neural_network import NeuralNetwork
import torch
from logic_gates import AND
from logic_gates import OR
from logic_gates import XOR
from logic_gates import NOT

'''
##### PART A #####
model = NeuralNetwork([2,2,5,6,3])
input_tensor= torch.tensor([[1,2,3],[3,4,5]], dtype = torch.double)
output = model.forward(input_tensor)
print(output)
'''

##### PART B #####
print("Results for AND:")
And = AND()
print(And(True, False))
'''print(And(False, True))
print(And(True, True))
print(And(False, False))

print("Results for OR:")
Or = OR()
print(Or(False,True))
print(Or(True,True))
print(Or(False,False))
print(Or(True,False))

print("Results for NOT:")
Not = NOT()
print(Not(False))
print(Not(True))

print("Results for XOR:")
Xor = XOR()
print(Xor(False,True))
print(Xor(False,False))
print(Xor(True,True))
print(Xor(True,False))
'''