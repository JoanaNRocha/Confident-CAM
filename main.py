# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:15:03 2023

@author: Joana Rocha
"""

import os
import pandas as pd

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import *
from ConfidentCAM import *

def main():
    
    #%% INPUT DATA
    
    # Select device    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ',device)
    
    # Select IMAGE and MODEL paths    
    img_path = 'F:/Multilabel_Joana/Main_Code/Git confident-cam/data/00000001_000.png'
    img = Image.open(img_path) # import image as numpy array
    
    model_path = 'F:/Multilabel_Joana/Main_Code/Git confident-cam/data/model_resnet50.pth' 
    th_path = 'F:/Multilabel_Joana/Main_Code/Git confident-cam/data/test_metrics_resnet50.xlsx' 
    
    labels = ['Atelectasis',
              'Cardiomegaly',
              'Effusion',
              'Infiltration',
              'Mass',
              'Nodule',
              'Pneumonia',
              'Pneumothorax',
              'Consolidation',
              'Edema',
              'Emphysema',
              'Fibrosis',
              'Pleural_Thickening',
              'Hernia']
    
    num_classes=len(labels)
    
    optimal_thresholds = pd.read_excel(th_path,usecols=[3],nrows=num_classes).to_numpy().flatten()
    
    model = my_Net(num_classes).to(device)
    model.load_state_dict(torch.load(model_path)) 
    model.eval()
    
    
    #%% CONFIDENT-CAM
    
    # get the equation parameters (m,b) necessary to find confidence score
    th_m_b_perclass = get_confidencescore_equationparameters(optimal_thresholds,labels)
    
    # get model prediction
    tensor_img,prediction = multilabel_classif_prediction(img,model,device)
    
    # get corresponding Grad-CAM for a certain class 'class_id' (can be replaced by any other 0-1 normalized XAI method)
    target_layers = [model.clmodel.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    
    class_id = 2 #'Effusion'
    targets = [ClassifierOutputTarget(class_id)]
    grayscale_cam = cam(input_tensor=tensor_img, targets=targets) #torch.Size([1, 3, 256, 256])
    grayscale_cam = grayscale_cam[0, :] 
    
    # get the Confident-CAM result based on the confidence score and the initial Grad-CAM   
    confidencescore,confident_CAM = ConfidentCAM(prediction,class_id,
                                                 th_m_b_perclass,labels,
                                                 grayscale_cam,
                                                 tensor_img)
    
    return 

#%%
if __name__ == '__main__':
    
    main()