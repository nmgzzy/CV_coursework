import math
import numpy as np

from MyConvolution import convolve

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma:
float) -> np.ndarray:
    lowKernel = makeGaussianKernel(lowSigma)
    highKernel = makeGaussianKernel(highSigma)
    lowpass = convolve(lowImage, lowKernel)
    lowpass2 = convolve(highImage, highKernel)
    lowpass = lowpass.astype(np.float32)
    lowpass2 = lowpass2.astype(np.float32)
    highpass = highImage - lowpass2
    hybrid = lowpass + highpass
    hmax = np.max(hybrid)
    hmin = np.min(hybrid)
    hybrid = (hybrid - hmin) / (hmax-hmin) * 255
    hybrid = hybrid.astype(np.uint8)
    return hybrid
 
def makeGaussianKernel(sigma: float) -> np.ndarray:
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