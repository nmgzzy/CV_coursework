import numpy as np
def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
# Convolve an image with a kernel assuming zero-padding of the image to handle the borders
# :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
# :type numpy.ndarray
# :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
# :type numpy.ndarray
# :returns the convolved image (of the same shape as the input image)
# :rtype numpy.ndarray

# Your code here. You'll need to vectorise your implementation 
# to ensure it runs at a reasonable speed.
    h = kernel.shape[0]
    w = kernel.shape[1]
    # invert kernel
    for i in range(h//2):
        for j in range(w):
            kernel[i][j], kernel[h-i-1][w-j-1] = kernel[h-i-1][w-j-1], kernel[i][j]
    if h%2 == 1:
        i = h//2
        for j in range(w//2):
            kernel[i][j], kernel[i][w-j-1] = kernel[i][w-j-1], kernel[i][j]
    # convolve image
    if len(image.shape) == 2:
        return _convolve(image, kernel, h, w)
    else:
        imgs = []
        for i in range(image.shape[2]):
            imgs.append(_convolve(image[:,:,i], kernel, h, w))
        dstack = np.dstack(imgs)
        return dstack

def _convolve(img, k, h, w):
    img_expend = np.pad(img, ((h, h), (w, w)), 'constant')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = np.sum(k * img_expend[i:i+k.shape[0], j:j+k.shape[1]])
            if img[i][j] > 255:
                img[i][j] = 255
            if img[i][j] < 0:
                img[i][j] = 0

    return img