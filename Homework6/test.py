import sys
import os
import cv2
import torch
import torchvision.transforms as transforms
import argparse


parser = argparse.ArgumentParser(description="Alexnet")
parser.add_argument('--model', type=str, help='path to directory of saved model')
args = parser.parse_args()
sys.argv = [sys.argv[0]]   
    
from train import alexnet

class test:
    def __init__(self):
        self.model = alexnet()
        path_to_model = os.path.join(args.model, 'alexnet_model.pth.tar')
        # load the trained model
        self.model = torch.load(path_to_model)
        
        
    def forward(self, img):        
        # make the byte tensor to float
        img = img.type(torch.FloatTensor)
        # change dimensions
        img = img.view(1,3,224,224)
        self.model.eval()
        # go through the forward pass
        output = self.model.forward_train(img)
        label = int(torch.max(output))
        label_str = self.model.val_classes[self.model.classname[label]]
        # return the string corresponding to the prediciton
        return label_str 
    
    
    def cam(self, index=0):
        # start the camera
        camera = cv2.VideoCapture(index)
        print("\nPress q to quit video capture\n")
        # camera window setting
        font = cv2.FONT_HERSHEY_SIMPLEX 
        camera.set(3, 1280)
        camera.set(4, 720)
        while True:
            # continuous frame capture
            retval, frame = camera.read()
            if retval:
                # resize the image to 224x224
                image = cv2.resize(frame, (224,224))
                # normalize, make into a float tensor, add extra dimension
                transform = transforms.Compose([transforms.ToPILImage(),transforms.Scale(256),transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                image = transform(image)
                image = image.unsqueeze(0)
                # forward pass to predict the label
                predicted_image = self.forward(image)
                # show the image
                cv2.putText(frame, predicted_image,(250, 50),font, 2, (255, 200, 100), 5, cv2.LINE_AA)
                cv2.imshow('Image', frame) 
                
            else:
                raise ValueError("Can't read frame")
                break
            key_press = cv2.waitKey(1)
            if key_press == ord('q'):
                break
            
        camera.release()
        cv2.destroyAllWindows() # Closing the window
        
      
if __name__ == '__main__':
    Test = test()
    Test.cam()
