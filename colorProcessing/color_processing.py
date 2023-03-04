from scipy import stats
import pandas as pd 
import cv2
import numpy as np
import joblib
import os
import glob


def rgb_to_hex(rgb_color):
    hex_color = '#'
    for i in rgb_color:
        hex_color +=("{:02x}".format(int(i)))
    return hex_color

def compSceneData(filePath):
    color = cv2.imread(filePath)
    colorRGB = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    colorHSV = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    rgb = colorRGB.reshape(1875*3000,3).astype(str)
    hsv = colorHSV.reshape(1875*3000,3).astype(str)
    hsv = list(pd.DataFrame(hsv).mode().iloc[0,:])
    rgb = list(pd.DataFrame(rgb).mode().iloc[0,:])
    colorHex = rgb_to_hex(rgb[1])
    data = [rgb[0],rgb[1],rgb[2],#RGB
           hsv[0],hsv[1],hsv[2],#HSV,
            colorHex,filePath.split('/')[-2],filePath.split('/')[-1]]
    return data

def applyDescriptor(cluster,color):
    if cluster == 1: return 'jewel'
    elif (cluster == 2) & (color in ['violet','red','blue']): return 'royal'
    elif cluster == 5: return 'muted'
    elif (cluster == 2) & (color in ['violet','yellow']): return 'dull'
    else:return

def main():
    svm = joblib.load('SVM_RGB_model.sav')
    kmeans = joblib.load('kMeans_S_model.sav')
    list_of_files = glob.glob('CNN_outs/')
    latest_file = max(list_of_files, key=os.path.getctime)
    
    data = compSceneData(latest_file)
    color = svm.predict(data[:3])
    cluster = kmeans.predict(data[4])
    descriptor =  applyDescriptor(cluster,color)
    res = ' '.join([cluster,descriptor])
    file = open("ColorProcess_outs/color.txt", "w")
    a = file.write(res)
    file.close()
    return res

if __name__ == "__main__":
    main()