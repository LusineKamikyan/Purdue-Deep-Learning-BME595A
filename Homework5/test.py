import torch
import torchvision
import torchvision.transforms as transforms
from img2num import img2num
from img2obj import img2obj


#Img2obj = img2obj()
#Img2obj.train()

#### PART A ####
Img2num = img2num()
Img2num.train()

###MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

test_set = torchvision.datasets.MNIST('/tmp', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
dataiter = iter(testloader)
images, data = dataiter.next()

counter = 0
for batch_index, (data, target) in enumerate(testloader):
    data = data.squeeze(0)
    data = data.squeeze(0)
    data = data.type(torch.ByteTensor)
    print(data.size())
    a = Img2num.forward(data) 
    if target == a:
        counter+=1



#### PART B ####

Img2obj = img2obj()
Img2obj.train()

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
    'worm'
]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

test_set = torchvision.datasets.CIFAR100('/tmp', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
dataiter = iter(testloader)
images, data = dataiter.next()

### CIFAR
counter = 0
for batch_index, (data, target) in enumerate(testloader):
    data = data.squeeze(0)
    #data = data.type(torch.ByteTensor)
    a = Img2obj.forward(data)
    if labels[target.item()] == a:
        counter+=1
    Img2obj.view(data)
            
