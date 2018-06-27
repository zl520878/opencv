#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 17:06
# @Author  : liangzhang7
# @Site    : 傅里叶变换
# @File    : fft.py
# @Desc    : 傅里叶变换


import cv2
import numpy as np
from scipy import ndimage

def fft():
    kernel_3x3 = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])

    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1,  1,  2,  1, -1],
                           [-1,  2,  4,  2, -1],
                           [-1,  1,  2,  1, -1],
                           [-1, -1, -1, -1, -1]])

    #参数0表示以灰度图形式打开图像
    img = cv2.imread("image/1.jpg",0)

    cv2.imshow("y", img)

    #创建3x3的核对图像进行卷积实现高通滤波
    k3 = ndimage.convolve(img, kernel_3x3)
    #创建3x3的核对图像进行卷积实现高通滤波
    k5 = ndimage.convolve(img, kernel_5x5)

    #先使用高斯模糊实现低通滤波，而后计算与原图像的差值，得到高通滤波效果
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    g_hpf = img - blurred

    cv2.imshow("3", blurred)
    # cv2.imshow("5", k5)
    cv2.imshow("g_hpf", g_hpf)

    cv2.waitKey()
    cv2.destroyAllWindows()


def edgeCheck():
    global ch
    img = cv2.imread("image/2.png")
    cv2.imshow("old", img)
    blur = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray, cv2.CV_8U, gray, 5)
    norm = (1.0 / 255) * (255 - gray)
    chan = cv2.split(img)
    for ch in chan:
        ch[:] = ch * norm
    zzz = cv2.merge(chan,img)
    cv2.imshow("new", img)

if __name__ == "__main__":
    #傅里叶变换
    #测试注释
    # fft()
    #边缘检测
    edgeCheck()

    cv2.waitKey()
    cv2.destroyAllWindows()