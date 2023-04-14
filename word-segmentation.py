import cv2
import numpy as np
import pandas as pd
import os
from time import time
OUTPUT_PATH = "SET OUTPUT PATH"
INPUT_IMAGE_PATH = "SET PATH TO INPUT DOCUMENT IMAGE"
img= cv2.imread(INPUT_IMAGE_PATH)
preprocess_start=time()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)

blur = cv2.GaussianBlur(thresh1,(1,1),0)
kernel = np.ones((1,1), np.uint8)

img_dilation = cv2.dilate(thresh1, kernel,iterations=2)
 
preprocess_stop = time()
word_segmentation_start=time()
contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_LIST  

 , cv2.CHAIN_APPROX_SIMPLE)

hierarchy=hierarchy[0]

print(hierarchy)

bounding_rects=[] 
bounding_rects_char=[]
i=0
filtered_contours=[]
for cnt in contours:
    print(str(i)+": "+ str(cv2.contourArea(cnt)))
    if(cv2.contourArea(cnt)<100):
        continue
    if(cv2.contourArea(cnt)>10000):
        continue
    filtered_contours.append(cnt)
    bounding_rects.append(cv2.boundingRect(cnt))
    i=i+1

i=0
# img_dilation = cv2.cvtColor(img_dilation, cv2.COLOR_GRAY2RGB)
for bound in bounding_rects:
    (x,y,w,h) = bound
    ROI = img[(y):(y+h), x:(x+w)]
    # cv2.rectangle(img_dilation, (x,y), (x+w, y+h), color=(0,255,0))
    os.mkdir(OUTPUT_PATH+"/"+str(i))
    word_img_path =OUTPUT_PATH+"/"+str(i)
    img_name =  word_img_path+"/ROI"+str(i) + ".png"
    cv2.imwrite(img_name, ROI)
    i=i+1  
word_segmentation_stop = time()
preprocessing_time = preprocess_stop-preprocess_start
word_segmentation_time = word_segmentation_stop- word_segmentation_start
# cv2.imshow("Contours", img_dilation)
# cv2.waitKey(0)
print(preprocessing_time)
print(word_segmentation_time)
