"""
RSI and RDI metrics
"""
# import metrics as met
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

"""
Function laplacian_of_gaussian_filter() is from
https://github.com/timoblak/OpenAFQA
"""
def laplacian_of_gaussian_filter(sigma):
    """Creates a 2D LOG filter based on the input sigma value

    :param sigma: Sigma to paramtrize the Gaussian
    :return: 2D filter
    """
    n = np.ceil(sigma*6)
    y, x = np.ogrid[-n//2:n//2+1, -n//2:n//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y)) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4))
    return final_filter

def LOG(block, sigma=1.6):
    # Calculates response for one sigma on image block (for dot detection)
    filter_log = laplacian_of_gaussian_filter(sigma)  # 500 ppi
    block_response = cv2.filter2D(block.astype(np.float64), -1, filter_log)  # convolving imag
    block_response = np.abs(block_response)
    return block_response

def spectrum(img, mode): 
    #fft
    H = np.fft.fft2(img)
    #shift (center is zero frequency)
    H_shift = np.fft.fftshift(H)
    #compute 2d spectrum (a+bi =>  sqrt(a^2+b^2))
    power_spectrum_2d = np.abs(H_shift)    
    
    # shape
    Heigh, Weight = power_spectrum_2d.shape
    # normalize 2d spectrum
    power_spectrum_2d_norm = power_spectrum_2d / (Heigh*Weight)
    if mode == 'mean':
        return np.mean(power_spectrum_2d_norm)
    if mode == 'max':
        return np.max(power_spectrum_2d_norm)


"""
RSI metrics
"""
def RSI_mean(img): 
    reponse = LOG(img, sigma=1.6)
    reponse_mean = spectrum(reponse, 'mean')
    return reponse_mean
    
def RSI_max(img): 
    reponse = LOG(img, sigma=1.6)
    reponse_max = spectrum(reponse, 'max')
    return reponse_max   
   
   
"""
RDI metrics is from
https://github.com/timoblak/OpenAFQA
"""    
def rdi_block(block):
    # Calculates response for one sigma on image block (for dot detection)
    filter_log = laplacian_of_gaussian_filter(1.6)  # 500 ppi
    block_response = cv2.filter2D(block.astype(np.float64), -1, filter_log)  # convolving imag
    lap_block = np.square(block_response)  # squaring the response
    return lap_block.max()    
    
