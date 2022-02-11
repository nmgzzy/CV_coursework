import numpy as np
import cv2
import os
import glob
import MyConvolution as mc
import time
import MyHybridImages as mh

def test1():
    imgpaths = glob.glob(os.path.join("./data/", "*.bmp"))
    src = cv2.imread(imgpaths[2], 0) #0) #cv2.IMREAD_UNCHANGED)
    src2 = cv2.imread(imgpaths[2], -1) #0) #cv2.IMREAD_UNCHANGED)
    print(f'src.shape {src.shape}')
    print(f'src2.shape {src2.shape}')
    cv2.imshow('src', src)
    cv2.imshow('src2', src2)
    m, n = 3, 4
    # k = np.array([[1]*n for _ in range(m)])
    k = np.array([[1,2,3],[4,5,6],[7,8,9]])
    k = k / np.sum(k)
    a = time.time()
    dst = mc.convolve(src, k)
    # dst2 = mc.convolve(src2, k)
    b = time.time()
    print(f'dst.shape {dst.shape}')
    # print(f'dst2.shape {dst2.shape}')
    print(f'卷积程序用时：{b-a}s')
    cv2.imshow('dst', dst)
    # cv2.imshow('dst2', dst2)
    cv2.waitKey(0)

def test2():
    a = mh.makeGaussianKernel(1)
    b = cv2.getGaussianKernel(9, 1)
    b = b * b.T
    print(a)
    print("======\n")
    print(b)
    print("======\n")
    print(np.sum(np.abs(a - b)) < 0.000001)
    
def test3():
    src1 = cv2.imread("./data/bird.bmp", cv2.IMREAD_UNCHANGED) #0) #cv2.IMREAD_UNCHANGED)
    src2 = cv2.imread("./data/plane.bmp", cv2.IMREAD_UNCHANGED) #0) #cv2.IMREAD_UNCHANGED)
    a = time.time()
    res = mh.myHybridImages(src2, 2, src1, 1)
    b = time.time()
    print(f'混合用时:{b-a}s')
    cv2.imshow('res', res)
    cv2.waitKey(0)

def main():
    # imgpaths = glob.glob(os.path.join("./data/", "*.bmp"))
    # for path in v:
    #     img = cv2.imread(path, 0)
    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)
    # a = time.time()
    # src = cv2.imread(imgpaths[2], 0) #0) #cv2.IMREAD_UNCHANGED)
    # blur = cv2.blur(src,(3,3))
    # b = time.time()
    # print(f'cv2卷积程序用时：{b-a}s')
    # cv2.imshow('src', src)
    # cv2.imshow('blur', blur)
    # cv2.waitKey(0)
    test3()
    
    
if __name__ == '__main__':
    main()