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

lower = []
upper = []


video=cv2.VideoCapture('http://100.84.40.82:8081')
dimDeviation = []
count = 0
maxC = 100

while True:
    success, img = video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # SETTING MASKS & CONTOUR GENERATION
    
    rmask = cv2.inRange(image, rlower, rupper)

    mask = rmask # by DEFAULT
    
    rcontours, hierarchy = cv2.findContours(rmask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    omask = cv2.inRange(image, olower, oupper)
    
    ocontours, hierarchy = cv2.findContours(omask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ymask = cv2.inRange(image, ylower, yupper)
    
    ycontours, hierarchy = cv2.findContours(ymask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    gmask = cv2.inRange(image, glower, gupper)
    
    gcontours, hierarchy = cv2.findContours(gmask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bmask = cv2.inRange(image, blower, bupper)
    
    bcontours, hierarchy = cv2.findContours(bmask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    imask = cv2.inRange(image, ilower, iupper)
    
    icontours, hierarchy = cv2.findContours(imask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vmask = cv2.inRange(image, vlower, vupper)
    
    vcontours, hierarchy = cv2.findContours(vmask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pmask = cv2.inRange(image, plower, pupper)

    pcontours, hierarchy = cv2.findContours(pmask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                            
    # TRACKING CONTOURS
    #                       
    bounders = []
    for contour in rcontours:
        bounders.append(contour)
    for contour in ocontours:
        bounders.append(contour)
    for contour in ycontours:
        bounders.append(contour)
    for contour in gcontours:
        bounders.append(contour)
    for contour in bcontours:
        bounders.append(contour)
    for contour in icontours:
        bounders.append(contour)
    for contour in vcontours:
        bounders.append(contour)
    for contour in pcontours:
        bounders.append(contour)                                                        
    
    dims = [0,0,0,0]

    # FINDING THE BIGGEST CONTOUR

    if len(bounders)!= 0:
        #the biggest contour minimum area is 500
        for contour in bounders:
            if count == 0:
                biggest_contour = max(bounders, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(biggest_contour)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),3)
            if cv2.contourArea(contour) > 800:
                  # find the biggest contour
                biggest_contour = max(bounders, key=cv2.contourArea)
                # draw a rectangle around the biggest contour
                x,y,w,h = cv2.boundingRect(biggest_contour)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),3)
                dims = [x,y,w,h]
                if len(rcontours) != 0:
                    for contour in rcontours[0][0]:

                        if contour[0] == biggest_contour[0][0][0] and contour[1] == biggest_contour[0][0][1]:
                                mask = rmask
                                lower = rlower
                                upper = rupper
                if len(ocontours) != 0:
                    for contour in ocontours[0][0]:
                        if contour[0] == biggest_contour[0][0][0] and contour[1] == biggest_contour[0][0][1]:
                                mask = omask
                                lower = olower
                                upper = oupper
                if len(ycontours) != 0:
                    for contour in ycontours[0][0]:
                        if contour[0] == biggest_contour[0][0][0] and contour[1] == biggest_contour[0][0][1]:
                                mask = ymask
                                lower = ylower
                                upper = yupper
                if len(gcontours) != 0:
                    for contour in gcontours[0][0]:
                        if contour[0] == biggest_contour[0][0][0] and contour[1] == biggest_contour[0][0][1]:
                                mask = gmask
                                lower = glower
                                upper = gupper
                if len(bcontours) != 0:
                    for contour in bcontours[0][0]:
                        if contour[0] == biggest_contour[0][0][0] and contour[1] == biggest_contour[0][0][1]:
                                mask = bmask
                                lower = blower
                                upper = bupper
                if len(icontours) != 0:
                    for contour in icontours[0][0]:
                        if contour[0] == biggest_contour[0][0][0] and contour[1] == biggest_contour[0][0][1]:
                                mask = imask
                                lower = ilower
                                upper = iupper
                if len(vcontours) != 0:
                    for contour in vcontours[0][0]:
                        if contour[0] == biggest_contour[0][0][0] and contour[1] == biggest_contour[0][0][1]:
                                mask = vmask
                                lower = vlower
                                upper = vupper
                if len(pcontours) != 0:
                    for contour in pcontours[0][0]:
                        if contour[0] == biggest_contour[0][0][0] and contour[1] == biggest_contour[0][0][1]:
                                mask = pmask
                                lower = plower
                                upper = pupper

                

                # convert the contour tuple to a numpy array
                # determine which colour the greatest contour is

                

    if len(bounders) == 0:
        count = 0
        dimDeviation = []
        continue
      
    # Object Detection with a live feed, static observation
    # x,y,w,h are within a similar range % for 5 seconds, then save an image within the rectangle
    
    dimDeviation.append(dims)
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

        a,b,c,d = 0,0,0,0
        # save the saved image as a subject, and read it's RGB values

        subj = cv2.imread('image.png')

        subj =  simplest_cb(subj,1)
        
        subject = cv2.cvtColor(subj, cv2.COLOR_BGR2HSV)

        if lower == [] or upper == []:
            count = 0
            dimDeviation = []
            continue

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
                       
       
        # reset the count and dimDeviation
        count = 0
        dimDeviation = []

        if subj.size == 0:
            count = 0
            dimDeviation = []
            continue
        # Save the image into the subject folder CNN_out
        
        cv2.imwrite('CNN_out/subject.png', subj)
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

