#THIS CODE TAKES PATH OF WORD IMAGE FILE AS INPUT AND GENERATES FILES OF IMAGES OF SEGMENTED CHARACTERS
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
def CharacterSegmentation(PATH):
    tolerance=5
    min_contour_area=10
    #Reading image
    img_path = os.path.join(PATH, os.listdir(PATH)[0])
    img= cv2.imread(img_path)
    og =img
    
    #____PREPROCESSING_________
    #Gray Scaling
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = img.shape[1]*87
    img_new=img.copy()
    shirolekha_idx=[]
    ##Whitening of shirolekha
    img_row_sum = np.sum(img,axis=1).tolist()
    plt.plot(img_row_sum)
    for i,row in enumerate(img):
        if(sum(img[i])<threshold):
            shirolekha_idx.append(i)
            img_new[i][:] =255
 
    ##thresholding  
    ret,thresh1 = cv2.threshold(img_new,100,255,cv2.THRESH_BINARY)
    ##applying gaussian blur
    blur = cv2.GaussianBlur(thresh1,(1,1),0)	
    ##image dilation
    kernel = np.ones((1,1), np.uint8)
    img_dilation = cv2.dilate(blur, kernel,iterations=1)
    # cv2.imshow("Processed", img_dilation)
    # cv2.waitKey(0)
    
    img_col_sum = np.sum(img_new,axis=0).tolist()
    #______CONTOURING________________-
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE  
    , cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    bounding_rects=[] 
    print("COMPONENTS:")
    i=0
    parent=-5
    hierarchy = hierarchy[0]
    new_contours=[]
    for component in zip(contours, hierarchy):
        # print(component)
        currentContour = component[0]
        currentHierarchy = component[1]
        print(str(i)+str(currentHierarchy))
        if(not currentHierarchy[3]<0):
            new_contours.append(currentContour)
        
        i+=1
    
    countoured=cv2.drawContours(og, new_contours, -1, (0,255,0), 1)
    # cv2.imshow("Contours: ", countoured)
    # cv2.waitKey(0)
    print(len(contours))
    print("END")    
    i=0
    for cnt in new_contours:
        print(str(cv2.contourArea(cnt)))
        if(cv2.contourArea(cnt)<min_contour_area):
            continue
        bounding_rects.append(cv2.boundingRect(cnt))
        i+=1

    bounding_rects = sorted(bounding_rects, key=lambda d: d[0]) 
    
    i=0
    for bound in bounding_rects:
        (x,y,w,h) = bound
        ROI = img[:, x:(x+w)]
        img_name =  PATH+"/"+str(i)+".png"
        cv2.imwrite(img_name, ROI)
        i=i+1
    



