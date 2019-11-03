import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import time
import argparse

#read the command line
parser = argparse.ArgumentParser() # (description="Alexnet")
parser.add_argument('--data', type=str) #, help='path to directory of tiny imagenet dataset')
parser.add_argument('--save', type=str) #, help='path to directory to save trained model')
args = parser.parse_args()
# trained data directory
fn_data = args.data
# saved data directory
fn_saved = args.save


class alexnet(nn.Module):
    def __init__(self):
        # define the CNN
        super(alexnet, self).__init__()
        self.__layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.__layer2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 200),
            )

         
    def forward_train(self, data):
        # move the image through the layers of the image
        data = self.__layer1(data)
        # make the data into 1d for FC layers
        data = data.view(-1,256*6*6)
        data = self.__layer2(data)
        return data    
    
    def Train(self):
        BATCH_SIZE = 100
        model = self
        
        # need to put the validation data in the right folder format
        path = os.path.join(fn_data, 'val/images')  
        # find the file 
        filename = os.path.join(fn_data, 'val/val_annotations.txt')
        #open it
        fp = open(filename, "r")  
        # read the lines
        data = fp.readlines()
        # creates a dictionary of validation images 
        val_img_dict = {}
        for line in data:
            strings = line.split("\t")
            val_img_dict[strings[0]] = strings[1]
        fp.close()
        # create the folders
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(path, folder))
            if not os.path.exists(newpath):  
                os.makedirs(newpath)
            if os.path.exists(os.path.join(path, img)):  
                os.rename(os.path.join(path, img), os.path.join(newpath, img))
                
        #create a dictionary of classes 
        # 'indices' in the dictionary will be the names n########
        filename = os.path.join(fn_data, 'words.txt') 
        fp = open(filename, "r")  
        data = fp.readlines()
        
        self.val_classes = {}
        for line in data:
            strings = line.split("\t")
            self.val_classes[strings[0]] = strings[1]
        fp.close()

            
        # load the data
        # citation: https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
        data_dir = fn_data
        TRAIN = 'train'
        VAL = 'val/images'
        
        data_transforms = {
            TRAIN: transforms.Compose([ 
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            VAL: transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }
        image_datasets = {
            x: torchvision.datasets.ImageFolder(
                os.path.join(data_dir, x), 
                transform=data_transforms[x]
            )
            for x in [TRAIN, VAL]
        }
        
        train_dataloader = torch.utils.data.DataLoader(image_datasets[TRAIN], batch_size = BATCH_SIZE, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(image_datasets[VAL], batch_size = BATCH_SIZE, shuffle = False) 

        train_data = image_datasets[TRAIN]
        self.classname = train_data.classes
        
        # copy the pretrained weights into the model
        # not the last layer
        ptrain_alexnet = models.alexnet(pretrained=True) 
        ptrain_alexnet.eval()
        
        # go through both of the models
        # copying the weight on all the layers except the last linear layer
        for i, j in zip(model.modules(), ptrain_alexnet.modules()):  
            if not list(i.children()):
                if len(i.state_dict()) > 0:  
                    if i.weight.size() == j.weight.size():  
                        i.weight.data = j.weight.data
                        i.bias.data = j.bias.data
                        
                        
        #freeze the layers
        for p in model.parameters():
            p.requires_grad = False
        # except the last one
        for p in model.__layer2[6].parameters():
            p.requires_grad = True
        
        # defining the epoch, and loss and time arrays
        epoch = 10
        L = np.zeros((epoch,1))
        TL = np.zeros((epoch,1))
        Epoch = np.zeros((epoch,1))
        Run_time = np.zeros((epoch,1))
        
        # optimizer and loss
        optimizer = torch.optim.Adam(model.__layer2[6].parameters(), lr=1e-3)
        CE_loss = torch.nn.CrossEntropyLoss()
        
        for i in range(epoch):
            print(i)
            Loss = 0
            start_time = time.time()
            model.train()
            for batch_index, (data, target) in enumerate(train_dataloader):
                #zero the gradient so it doesn't accumulate
                optimizer.zero_grad()
                
                print(batch_index)
                # forward pass through CNN
                print(data.size())
                
                output_forward = self.forward_train(data)
                print('passed forward')
                # calculate the CE loss
                loss = CE_loss(output_forward, target)
                # add the losses uo to each other
                Loss=Loss+loss
                print(Loss)
                
                # backward pass
                loss.backward()
                print('passed backward')
                #update the parmameters
                optimizer.step()
                torch.save(an,os.path.join(fn_saved, 'alexnet_model.pth.tar'))

               
            # average the losses
            L[i] = Loss.detach().numpy()/1000
            print(L[i])
            Loss = 0 
            Epoch[i]=i
            model.eval()
            for batch_index, (data, target) in enumerate(test_dataloader):
                print(batch_index)
                # forward pass through CNN
                output_forward = self.forward_train(data)
                # calculate the CE loss
                loss = CE_loss(output_forward, target)                
                # add looses to each other
                Loss=Loss+loss
            end_time = time.time()
            # average losses
            TL[i] = Loss.detach().numpy()/100
            print(TL[i])
            # time to run through test and train
            Run_time[i] = end_time-start_time
            
            torch.save(an,os.path.join(fn_saved, 'alexnet_model.pth.tar'))

            
        plt.figure(0)
        plt.plot(Epoch,L)
        plt.plot(Epoch,TL)
        plt.figure(1)
        plt.plot(Epoch, Run_time)
                
        
if __name__ == '__main__':
    an = alexnet()
    an.Train()
