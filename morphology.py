import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pun.jpg',0) 
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
outline = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(erosion,cmap = 'gray')
plt.title('Erosion'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(dilation,cmap = 'gray')
plt.title('Dilation'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(opening,cmap = 'gray')
plt.title('Opening'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(closing,cmap = 'gray')
plt.title('Closing'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(outline,cmap = 'gray')
plt.title('Outline'), plt.xticks([]), plt.yticks([])

plt.show()
