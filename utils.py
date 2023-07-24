# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:59:14 2023

@author: Joana Rocha
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image

#%% MODEL

class my_Net(nn.Module):
    '''
    Multi-label classification model architecture.
    '''
    
    def __init__(self, num_classes):
        super(my_Net, self).__init__()
        
        self.num_classes = num_classes
        
        #ResNet50
        self.clmodel = models.resnet50(weights="IMAGENET1K_V1")
        self.clmodel.fc = nn.Linear(2048, self.num_classes)

    def forward(self,x):
        h=self.clmodel(x)
        return torch.sigmoid(h)  

def multilabel_classif_prediction(img,model,device):
    '''
    Predicts the output probability for a certain image, based on the given model. 
    
    Inputs:
        img: PIL image whose classes we seek to predict.
        model: pre-trained loaded model.
        device: selected device.

    Outputs:
        img: tensor image.
        prediction: array containing the probability of the N classes.
    '''
    
    # preprocess image - adapt according to pre-trained model being used
    transform = transforms.Compose([transforms.Resize((256,256)), 
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], #ImageNet mean values 
                                                          std=[0.229, 0.224, 0.225]) #ImageNet std values 
                                    ])
    img = transform(img).to(device).unsqueeze(0)
    
    # make prediction for all classes
    prediction = model(img)
    prediction = prediction.cpu().detach().numpy()
    
    return img,prediction.T.flatten()
