#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/10 14:22
# @Author  : liangzhang7
# @Site    : 
# @File    : main.py
# @Software: PyCharm

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("image/2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray,)
cv2.imshow("lines", gray)
cv2.waitKey()
edges = cv2.Canny(gray, 20, 180)
minLineLength = 20
maxLineGap = 20
lines = cv2.HoughLinesP(edges, 3, np.pi/180, 100, minLineLength, maxLineGap)

# for x1, y1, x2, y2 in lines[0]:
#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

# cv2.imshow("edges", edges)
# cv2.imshow("lines", image)
# cv2.waitKey()
cv2.destroyAllWindows()