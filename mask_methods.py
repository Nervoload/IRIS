import cv2
import numpy as np
def masker(image, lower, upper, count, env):

    mask = cv2.inRange(image, lower, upper)
 
    contours, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x,y,w,h = 0,0,0,0
    
    if len(contours)!= 0:
        #the biggest contour minimum area is 500
        for contour in contours:
            if count == 0:
                biggest_contour = max(contours, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(biggest_contour)
                cv2.rectangle(env, (x,y), (x+w,y+h), (0,0,255),3)
            if cv2.contourArea(contour) >1000:
                  # find the biggest contour
                biggest_contour = max(contours, key=cv2.contourArea)
                # draw a rectangle around the biggest contour
                x,y,w,h = cv2.boundingRect(biggest_contour)

    return [[x,y,w,h], mask]

def dimCalc(dimensions):
    if len(dimensions) == 0:
        return 0
    area = (dimensions[0] + dimensions[2]) * (dimensions[1] + dimensions[3])
    return area