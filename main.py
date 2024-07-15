import cv2
import numpy as np

img = np.ones((350, 700, 3), dtype = np.uint8)
img = 255* img

cv2.imshow('white image', img)
cv2.waitKey(0)  