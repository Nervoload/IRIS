import cv2
import numpy as np
from color_correction import simplest_cb
from mask_methods import masker, dimCalc

rlower = np.array([255,0,0])
rupper = np.array([255,50,0])
olower = np.array([255,51,0])
oupper = np.array([255,170,0])
ylower = np.array([255,171,0])
yupper = np.array([242,255,0])
glower = np.array([241,255,0])
gupper = np.array([0,255,119])
blower = np.array([0,255,120])
bupper = np.array([0,95,255])
ilower = np.array([0,94,255])
iupper = np.array([50,0,255])
vlower = np.array([51,0,255])
vupper = np.array([255,0,255])
plower = np.array([255,0,254])
pupper = np.array([255,0,1])

lower = []
upper = []

video=cv2.VideoCapture(0)
dimDeviation = []
count = 0
maxC = 100

while True:
    maxMask = np.array([])
    success, img = video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # detect only one object at a time
    
    rCo = masker(image,rlower,rupper,count,img)
    oCo = masker(image,olower,oupper,count,img)
    yCo = masker(image,ylower,yupper,count,img)
    gCo = masker(image,glower,gupper,count,img)
    bCo = masker(image,blower,bupper,count,img)
    iCo = masker(image,ilower,iupper,count,img)
    vCo = masker(image,vlower,vupper,count,img)
    pCo = masker(image,plower,pupper,count,img)

    rMask = rCo[1]
    oMask = oCo[1]
    yMask = yCo[1]
    gMask = gCo[1]
    bMask = bCo[1]
    iMask = iCo[1]
    vMask = vCo[1]
    pMask = pCo[1]
    
    
    coArr  = [rCo[0],oCo[0],yCo[0],gCo[0],bCo[0],iCo[0],vCo[0],pCo[0]]

    dimArr = [dimCalc(rCo[0]),dimCalc(oCo[0]),dimCalc(yCo[0]),dimCalc(gCo[0]),dimCalc(bCo[0]),dimCalc(iCo[0]),dimCalc(vCo[0]),dimCalc(pCo[0])]
  
    maxDim = max(dimArr)

    maxIndex = dimArr.index(maxDim)

    maxCo = coArr[maxIndex]


    if maxCo == rCo:
        lower = rlower
        upper = rupper
        maxMask = rMask
    elif maxCo == oCo:
        lower = olower
        upper = oupper
        maxMask = oMask
    elif maxCo == yCo:
        lower = ylower
        upper = yupper
        maxMask = yMask
    elif maxCo == gCo:
        lower = glower
        upper = gupper
        maxMask = gMask
    elif maxCo == bCo:
        lower = blower
        upper = bupper
        maxMask = bMask
    elif maxCo == iCo:
        lower = ilower
        upper = iupper
        maxMask = iMask
    elif maxCo == vCo:
        lower = vlower
        upper = vupper
        maxMask = vMask
    elif maxCo == pCo:
        lower = plower
        upper = pupper
        maxMask = pMask


    
    cv2.rectangle(img, (maxCo[0],maxCo[1]), (maxCo[0]+maxCo[2],maxCo[1]+maxCo[3]), (0,0,255),3)

      
    # Object Detection with a live feed, static observation
    # x,y,w,h are within a similar range % for 5 seconds, then save an image within the rectangle
    
    dimDeviation.append(maxCo)

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

        # turn lower and upper to numpy arrays
        lower = np.array(lower)
        upper = np.array(upper)

        subMask = cv2.inRange(subject, lower, upper)
        
        subContours, newHierarchy = cv2.findContours(subMask,
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(subContours)!= 0:
            #the biggest contour minimum area is 500
            for contour in subContours:
                if cv2.contourArea(contour) >1000:
                    # find the biggest contour
                    subject_contour = max(subContours, key=cv2.contourArea)
                    # draw a rectangle around the biggest contour
                    a,b,c,d = cv2.boundingRect(subject_contour)
                    cv2.rectangle(subj, (a,b), (a+c,b+d), (0,0,255),3)
                    subCo = [a,b,c,d]
        if len(subContours) == 0:
            continue



        # Save the image into the subject folder CNN_outs
        
        subj = subj[subCo[1]:subCo[1]+subCo[3], subCo[0]:subCo[0]+subCo[2]]
        cv2.imwrite('CNN_outs/subject.png', subj)

        # Classify the image with the SVM


        # reset the count and dimDeviation
        count = 0
        dimDeviation = []

    # show the webcam and mask

    if maxMask.size == 0:
        continue

    cv2.imshow('mask',maxMask)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)

    # exit conditions with esc
    if cv2.waitKey(1) == 27:
        # reset the count and dimDeviation
        count = 0
        dimDeviation = []
        break

