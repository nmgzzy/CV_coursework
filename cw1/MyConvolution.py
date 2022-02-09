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

    if len(image.shape) == 2:
        return _convolve(image, kernel)
    else:
        imgs = []
        for i in range(image.shape[2]):
            imgs.append(_convolve(image[:,:,i], kernel))
        dstack = np.dstack(imgs)
        return dstack

def _convolve(img, k):
    h = k.shape[0] // 2
    w = k.shape[1] // 2
    for i in range(h):
        for j in range(w):
            k[i][j], k[h-i-1][w-j-1] = k[h-i-1][w-j-1], k[i][j]
    img_expend = np.pad(img, ((h, h), (w, w)), 'constant')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = k 
    return img