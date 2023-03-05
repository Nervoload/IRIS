import cv2
import numpy as np
from color_correction import simplest_cb

rlower = np.array([0,150,150])
rupper = np.array([10,255,255])
olower = np.array([11,150,150])
oupper = np.array([15,255,255])
ylower = np.array([16,171,128])
yupper = np.array([39,255,255])
glower = np.array([40,150,20])
gupper = np.array([80,255,255])
blower = np.array([81,128,120])
bupper = np.array([105,255,255])
ilower = np.array([106,128,120])
iupper = np.array([120,255,255])
vlower = np.array([121,128,120])
vupper = np.array([145,255,255])
plower = np.array([146,128,120])
pupper = np.array([168,255,255])


#lower = rupper
#upper = rlower

lower = glower
upper = gupper



video=cv2.VideoCapture(0)
dimDeviation = []
count = 0
maxC = 100

while True:
    success, img = video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # detect only one object at a time
    
    mask = cv2.inRange(image, lower, upper)
    
    contours, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours)!= 0:
        #the biggest contour minimum area is 500
        for contour in contours:
            if count == 0:
                biggest_contour = max(contours, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(biggest_contour)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),3)
            if cv2.contourArea(contour) > 800:
                  # find the biggest contour
                biggest_contour = max(contours, key=cv2.contourArea)
                # draw a rectangle around the biggest contour
                x,y,w,h = cv2.boundingRect(biggest_contour)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),3)
    if len(contours) == 0:
        
        continue
      
    # Object Detection with a live feed, static observation
    # x,y,w,h are within a similar range % for 5 seconds, then save an image within the rectangle
    
    dimDeviation.append([x,y,w,h])
    count += 1
    if count == maxC:
        # make dimDeviation a numpy array
        dimDeviation = np.array(dimDeviation)
        # find the standard deviation of each column in dimDeviation
        dimStd = np.std(dimDeviation, axis=0)
        print(dimStd)
        # if the standard deviation is less than 10, then save the current rectangle as an image
        if float(dimStd[0]) < 200 and float(dimStd[1]) < 200 and float(dimStd[2]) < 100 and float(dimStd[3]) < 100:
            # save the entire screen as an image

            cv2.imwrite('image.png', img)


        # Subject Detection & image colour correction


        # save the saved image as a subject, and read it's RGB values

        subj = cv2.imread('image.png')

        subj =  simplest_cb(subj,1)
        
        subject = cv2.cvtColor(subj, cv2.COLOR_BGR2HSV)

        subMask = cv2.inRange(subject, lower, upper)
        
        subContours, newHierarchy = cv2.findContours(subMask,
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(subContours)!= 0:
            #the biggest contour minimum area is 500
            for contour in subContours:
                if cv2.contourArea(contour) >800:
                    # find the biggest contour
                    subject_contour = max(subContours, key=cv2.contourArea)
                    # draw a rectangle around the biggest contour
                    a,b,c,d = cv2.boundingRect(subject_contour)
                    cv2.rectangle(subj, (a,b), (a+c,b+d), (0,0,255),3)
                    subj = subj[b:b+d, a:a+c]
        if len(subContours) == 0:
            continue
        

        # Save the image into the subject folder CNN_outs
        if subj.size != 0:
            cv2.imwrite('CNN_outs/subject.png', subj)

        # Classify the image with the SVM


        # reset the count and dimDeviation
        count = 0
        dimDeviation = []

    # show the webcam and mask

    cv2.imshow('mask',mask)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)

    # exit conditions with esc
    if cv2.waitKey(1) == 27:
        # reset the count and dimDeviation
        count = 0
        dimDeviation = []
        break

