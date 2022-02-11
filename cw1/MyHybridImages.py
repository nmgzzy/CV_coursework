import math
import numpy as np
from MyConvolution import convolve
def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma:
float) -> np.ndarray:
    #  """
    #  Create hybrid images by combining a low-pass and high-pass filtered pair.
    
    #  :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or
    # colour shape=(rows,cols,channels))
    #  :type numpy.ndarray
    
    #  :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering
    # lowImage
    #  :type float
    
    #  :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or
    # colour shape=(rows,cols,channels))
    #  :type numpy.ndarray
    
    #  :param highSigma: the standard deviation of the Gaussian used for low-pass filtering
    # highImage before subtraction to create the high-pass filtered image
    #  :type float
    
    #  :returns returns the hybrid image created
    #  by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it
    # with
    #  a high-pass image created by subtracting highImage from highImage convolved with
    #  a Gaussian of s.d. highSigma. The resultant image has the same size as the input
    # images.
    #  :rtype numpy.ndarray
    #  """
    # Your code here.
    lowKernel = makeGaussianKernel(lowSigma)
    highKernel = makeGaussianKernel(highSigma)
    lowpass = convolve(lowImage, lowKernel)
    highpass = highImage - convolve(highImage, highKernel)
    return lowpass + highpass
 
def makeGaussianKernel(sigma: float) -> np.ndarray:
    #  """
    #  Use this function to create a 2D gaussian kernel with standard deviation sigma.
    #  The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    #  floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    #  """
    # Your code here.
    t = int(8*sigma+1)
    n = t if t % 2 == 1 else t+1
    kernel = np.zeros((n,n), dtype=float)
    r = n // 2
    a = 1 / (2 * np.pi * sigma**2)
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            kernel[i+r][j+r] = a * np.exp(-(i**2 + j**2) / (2*sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel