import numpy as np
import cv2
import os
import glob
import MyConvolution as mc
import time

def main():
    imgs = glob.glob(os.path.join("./data/", "*.bmp"))
    # for path in imgs:
    #     img = cv2.imread(path, 0)
    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)

    img = cv2.imread(imgs[2], cv2.IMREAD_UNCHANGED) #0) #cv2.IMREAD_UNCHANGED)
    # blur = cv2.blur(img,(5,5))
    # img = img - blur
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    a = 1/9
    k = np.array([[a]*3 for _ in range(3)])
    a = time.time()
    img = mc.convolve(img, k)
    b = time.time()
    print(f'卷积程序用时：{b-a}s')
    cv2.imshow('image', img)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main()