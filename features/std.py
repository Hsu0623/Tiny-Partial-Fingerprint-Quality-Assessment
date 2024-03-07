import cv2
import os
import numpy as np

def none(block):
    assert block.dtype == 'uint8', 'the dtype of block to compute std should be uint8'
    return np.mean(block)/255
    
        