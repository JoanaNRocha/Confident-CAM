# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:34:35 2023

@author: Joana Rocha
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

#%% CONFIDENT-CAM

def get_confidencescore_equationparameters(optimal_thresholds,labels):
    '''
    Calculates the slope and y-intercept values for the Confidence Score rescaling equation, 
    based on the optimal threshold value per class.
    
    Inputs:
        optimal_thresholds: array with the optimal threshold value per class (dimension equal to the number of classes, N).
        labels: list containing the pathology names in the same order as the optimal thresholds (dimension equal to the number of classes, N).

    Outputs:
        th_m_b_perclass: dictionary containing N lists. Each list stores the class threshold, slop, and y-intercept values.
    '''
    
    th_m_b_perclass = {k: [] for k in labels}
    
    for pat in range(0,len(optimal_thresholds)):
        th=optimal_thresholds[pat]
        m=1/(1-th)
        b=1-m
        th_m_b_perclass[labels[pat]]=[th,m,b]
    
    return th_m_b_perclass


def calculate_confidencescore(prediction,class_id,th_m_b_perclass,labels):
    '''
    Calculates the confidence score of an instance for a specific class, 
    based on the optimal threshold value and probability output of that class.
    
    Inputs:
        prediction: array containing the probability of the N classes.
        class_id: number of the class to analyse.
        th_m_b_perclass: dictionary containing N lists. Each list stores the class threshold, slop, and y-intercept values.
        labels: list containing the pathology names in the same order as the optimal thresholds (dimension equal to the number of classes, N).

    Outputs:
        confidencescore: float/scalar storing the confidence score of a specific instance, for a specific class.
    '''
    
    confidencescore = th_m_b_perclass[labels[class_id]][1] * prediction[class_id] + th_m_b_perclass[labels[class_id]][2]  #m*ypred+b 
    confidencescore = max(confidencescore,0)
    
    return confidencescore


def ConfidentCAM(prediction,class_id,
                 th_m_b_perclass,labels,
                 XAImap,
                 tensor_img=None):
    '''
    Calculates the confidence score of an instance for a specific class, and 
    the corresponding Confident-CAM based on the provided initial XAI map.
    
    Inputs:
        prediction: array containing the probability of the N classes.
        class_id: number of the class to analyse.
        th_m_b_perclass: dictionary containing N lists. Each list stores the class threshold, slop, and y-intercept values.
        labels: list containing the pathology names in the same order as the optimal thresholds (dimension equal to the number of classes, N).
        XAImap: original CAM array of size (image width, image height). Any 0-1 normalized method can be used here (Grad-CAM, etc).
        tensor_img: original image array of size (image width, image height). If provided, the Confident-CAM results will be plotted. 

    Outputs:
        confidencescore: float/scalar storing the confidence score of a specific instance, for a specific class.
        confident_CAM: resulting map array of size (image width, image height).
    '''
    
    confidencescore = calculate_confidencescore(prediction,class_id,th_m_b_perclass,labels)
    confident_CAM = XAImap * confidencescore
    
    if tensor_img is not None:
        
        # convert a Tensor to numpy image
        tensor_img = torch.squeeze(tensor_img).cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406]) #ImageNet mean values 
        std = np.array([0.229, 0.224, 0.225]) #ImageNet std values
        tensor_img = std * tensor_img + mean
        tensor_img = np.clip(tensor_img, 0, 1)
        
        # generate CAM visualization, overlapped on original image
        heatmap = cv2.applyColorMap(np.uint8(255 * XAImap), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        CAM_onimg = heatmap + tensor_img
        CAM_onimg = CAM_onimg / np.max(CAM_onimg)
        CAM_onimg = np.uint8(255 * CAM_onimg)
        
        # generate Confident-CAM visualization, overlapped on original image
        heatmap = cv2.applyColorMap(np.uint8(255 * confident_CAM), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        confident_CAM_onimg = heatmap + tensor_img
        confident_CAM_onimg = confident_CAM_onimg / np.max(confident_CAM_onimg)
        confident_CAM_onimg = np.uint8(255 * confident_CAM_onimg)
        
        # plot original image, CAM, and Confident-CAM
        fig = plt.figure(figsize=(10,5)) 
        grid = ImageGrid(fig, 111,
                        nrows_ncols = (1,3),
                        axes_pad = 0.05,
                        cbar_location = "right",
                        cbar_mode="single",
                        cbar_size="5%",
                        cbar_pad=0.05
                        )
        
        grid[0].grid(False)
        grid[0].axis('off')
        grid[1].grid(False)
        grid[1].axis('off')
        grid[2].grid(False)
        grid[2].axis('off')
        
        grid[0].title.set_text('Original Image')
        grid[0].imshow(tensor_img,cmap='gray')
        
        grid[1].title.set_text('CAM')
        visualization = np.array(cv2.cvtColor(CAM_onimg, cv2.COLOR_BGR2RGB))
        grid[1].imshow(visualization)   
        
        grid[2].title.set_text('Confident-CAM')
        weighted_visualization = np.array(cv2.cvtColor(confident_CAM_onimg, cv2.COLOR_BGR2RGB))
        grid[2].imshow(weighted_visualization)  
        
        #add colormap of CAM
        plt.colorbar(plt.imshow(XAImap,cmap='jet'), cax=grid.cbar_axes[0], orientation='vertical',ticks=[0,0.5,XAImap.max()])
        
        if confident_CAM.max()<0.5:
            XAI_text = 'Warning: the confidence regarding this diagnosis is under 50%. Either the predicted pathology \nor its corresponding location could be wrong.'
            plt.figtext(0, 0, XAI_text, fontsize = 15) 
            
        plt.suptitle('Pathology: ' + labels[class_id] +'\n Confidence Score: ' + str(round(confidencescore,3)) + ', Probability: ' + str(round(prediction[class_id],3)) + ', Threshold: ' + str(round(th_m_b_perclass[labels[class_id]][0],3)))
        
    return confidencescore,confident_CAM



    