"""
The code of OCL metrics is sourced from the following references:

T. Oblak, R. Haraksim, P. Peer, L. Beslay. 
Fingermark quality assessment framework with classic and deep learning ensemble models. 
Knowledge-Based Systems, Volume 250, 2022    

T. Oblak, R. Haraksim, L. Beslay, P. Peer. 
Fingermark Quality Assessment: An Open-Source Toolbox. 
In proceedings of the International Conference of the Biometrics Special Interest Group (BIOSIG), pp. 159-170, 2021.

https://github.com/timoblak/OpenAFQA
"""

import cv2
import os
import numpy as np

def covcoef(img):
    
    assert len(img.shape) == 2, 'input img should gray.'

    # Central differences with filter2D
    kernelx = np.array([[-0.5, 0, 0.5]])
    kernely = np.array([[-0.5], [0], [0.5]])

    # calculate central differences
    fx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    fy = cv2.filter2D(img, cv2.CV_64F, kernely)
        
    cova = np.mean(fx * fx)
    covb = np.mean(fy * fy)
    covc = np.mean(fx * fy)

    return cova, covb, covc

def ocl(img):
    a, b, c = covcoef(img)
    eigvmax = ((a + b) + np.sqrt((a - b)*(a - b) + 4 * c * c)) / 2
    eigvmin = ((a + b) - np.sqrt((a - b)*(a - b) + 4 * c * c)) / 2
    if eigvmax != 0: 
        return 1-(eigvmin / eigvmax)  # return 1-[1(worst) - 0(best)] 
    else:
        return 0