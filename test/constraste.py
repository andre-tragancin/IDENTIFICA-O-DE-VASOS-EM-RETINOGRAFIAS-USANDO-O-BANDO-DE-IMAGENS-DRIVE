# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from skimage import io,exposure

#img = cv2.imread('gcnImgRGB.png',0)
img = io.imread("mask/01_test_mask.gif")
#img = io.imread("2nd_manual/01_manual2.gif")

#(b,g,r) = cv2.split(img)

#img = g
'''
imgAlargada = np.zeros((img.shape[0], img.shape[1]), np.uint8)

fmax = 0
fmin = 256

for x in range(0, img.shape[1]):
    for y in range(0, img.shape[0]):
        if(fmax < img[y][x]):
            fmax = img[y][x]
        if fmin > img[y][x]:
            fmin = img[y][x]
a = 255.0/(fmax - fmin)
b = -a*fmin

for x in range(0, img.shape[1]):
    for y in range(0, img.shape[0]):
        imgAlargada[y][x] = a*img[y][x]+b
'''
plt.hist(img.ravel(),256,[0,256], label='Original')
#plt.hist(imgAlargada.ravel(),256,[0,256], label='Alargada')

#plt.legend()
plt.show()

exposure.histogram(img, nbins=2)

#cv2.imshow('Imgem Original', img)
#cv2.imshow('Imgem Alargada', imgAlargada)
#cv2.imwrite('gcnImgRGBAlargada2.png',imgAlargada)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
