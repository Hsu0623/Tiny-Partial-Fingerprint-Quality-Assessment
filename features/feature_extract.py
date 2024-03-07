from .std import none
from .rsi_rdi import RSI_max, RSI_mean, rdi_block
from .ocl import ocl
import numpy as np


def get_block_info(img, metric, num_block_h=5, num_block_w=3):
    """
    Calculates metric's values for sliding window processing
    - one img can be divide into 2*(num_block_h*num_block_w) blocks
      - divide from top left to bottom right and from bottom right to top left 
      - single block_height=img.size(0)//num_block_h
      - single block_width=img.size(1)//num_block_w
    
    :param img: an fingerprint gray image (uint8)
    :param metric: one metrics function
    :param num_block_h: 
    :param num_block_w: 
    :return: mean and std
    """
    
    H, W = img.shape
    block_h = H // num_block_h
    block_w = W // num_block_w
    
    blocks_value = []
    
    # start from top left
    for i in range(num_block_h):
        for j in range(num_block_w):
            block = img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            value = metric(block)
            blocks_value.append(value)
    
    # start from bottom right    
    for i in range(num_block_h):
        for j in range(num_block_w):
            block = img[H-(i+1)*block_h:H-i*block_h, W-(j+1)*block_w:W-j*block_w]
            value = metric(block)
            blocks_value.append(value)
    
    blocks_value = np.array(blocks_value)
    
    return np.mean(blocks_value), np.std(blocks_value)
    

def get_features(img, num_block_h=5, num_block_w=3):
    """ 
    compute feature of image
    - block rsi 
        - rsi_mean and rsi_max
    - block std (b_std) 
        **small is better**
    - block ocl (b_ocl)
    - block rdi
    """
    
    rsi_mean, _ = get_block_info(img, RSI_mean, num_block_h, num_block_w)
    rsi_max, _ = get_block_info(img, RSI_max, num_block_h, num_block_w)
    _, b_std = get_block_info(img, none, num_block_h, num_block_w)
    b_ocl, _ = get_block_info(img, ocl, num_block_h, num_block_w)
    rdi, _ = get_block_info(img, rdi_block, num_block_h, num_block_w)
    
    return rsi_mean, rsi_max, b_std, b_ocl, rdi
    
    
    