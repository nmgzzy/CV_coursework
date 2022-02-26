import numpy as np
def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = kernel.shape
    hr, wr = h//2, w//2
    # invert kernel
    for i in range(hr):
        for j in range(w):
            kernel[i][j], kernel[h-i-1][w-j-1] = kernel[h-i-1][w-j-1], kernel[i][j]
    if h%2 == 1:
        i = hr
        for j in range(wr):
            kernel[i][j], kernel[i][w-j-1] = kernel[i][w-j-1], kernel[i][j]
    # convolve image
    ret = np.zeros(image.shape, dtype=image.dtype)
    if len(image.shape) == 2:
        ch = 1
        img_expend = np.pad(image, ((hr, hr), (wr, wr)), 'constant')
    else:
        ch = image.shape[2]
        img_expend = np.pad(image, ((hr, hr), (wr, wr), (0, 0)), 'constant')
    for c in range(ch):
        for i in range(ret.shape[0]):
            for j in range(ret.shape[1]):
                if ch == 1:
                    ret[i][j] = np.sum(kernel * img_expend[i:i+h, j:j+w])
                else:
                    ret[i][j][c] = np.sum(kernel * img_expend[i:i+h, j:j+w, c])
    return ret
