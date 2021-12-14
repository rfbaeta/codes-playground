import cv2
import numpy as np


img_extract_mean = cv2.imread("./data/extract_mean.png",1)
img_apply_mean = cv2.imread("./data/apply_mean.png",1)

b = img_extract_mean[:,:,0].mean()
g = img_extract_mean[:,:,1].mean()
r = img_extract_mean[:,:,2].mean()


#print(b)
#print(g)
#print(r)

b1 = img_apply_mean[:,:,0].mean()
g1 = img_apply_mean[:,:,1].mean()
r1 = img_apply_mean[:,:,2].mean()


equ1 = cv2.equalizeHist(img_apply_mean[:,:,0])
equ2= cv2.equalizeHist(img_apply_mean[:,:,1])
equ3 = cv2.equalizeHist(img_apply_mean[:,:,2])

a = np.zeros(img_apply_mean.shape)
a[:,:,0] = equ1
a[:,:,1] = equ2
a[:,:,2] = equ3
cv2.imwrite("./teste2.png",a)

difB = (b - b1)
difR = (r-r1)
difG = (g - g1)


img_apply_mean[:,:,0] = img_apply_mean[:,:,0] + difB + 10
img_apply_mean[:,:,1] = img_apply_mean[:,:,1] + difG + 10
img_apply_mean[:,:,2] = img_apply_mean[:,:,2] + difR + 10

equ1 = cv2.equalizeHist(img_apply_mean[:,:,0])
equ2= cv2.equalizeHist(img_apply_mean[:,:,1])
equ3 = cv2.equalizeHist(img_apply_mean[:,:,2])
a[:,:,0] = equ1
a[:,:,1] = equ2
a[:,:,2] = equ3

cv2.imwrite("./teste3.png",a)

