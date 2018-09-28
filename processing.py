# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:57:52 2018

@author: vinay
"""
import cv2
import numpy as np
from Main import main
import sympy as sp
import traceback
import re
from sympy import solveset
from sympy import Symbol
#from skimage.feature import hog
symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9","b", "-","/", "(","m", "p", "+", ")", "*", "w", "y"]
specialchars = ["-", "(", ")", "+", "*", "/"]
variables = ["m", "p", "w", "y", "b"]

def sort(iList, xList, yList):
    indexList = np.argsort(xList)
    sortedimgList = []
    for i in range(len(indexList)):
        if xList[indexList[i]]==0:
            continue
        if i!=0: 
            if (abs(xList[indexList[i-1]]-xList[indexList[i]])<=10) and (abs(yList[indexList[i-1]]-yList[indexList[i]])<=10):
                continue
            if yList[indexList[i]]<100 and yList[indexList[i]]>500:
                continue
        sortedimgList.append(iList[indexList[i]])
    return sortedimgList
#Check balanced parantheses
def eolCheck(string):
    sum = 0
    for i in range(len(string)):
        if string[i]=="(":
            sum +=1
        if string[i]==")":
            sum -=1
        if sum<0:
            return False
    if sum != 0:
        return False
    #check for cases like "2+" or "2+-2" or "+2*"
    #split = re.split('\+|\*|\-', string)
    #for i in range(len(split)):
    #    if not split[i]:
    #        return False
    return True
            

def checkInt(string):
    if str(int(string)) == string:
        return True
    else:
        return False
#Check for 02
def invalidTokenCheck(string):
    proper = ""
    temp = ""
    for i in range(len(string)):
        print(temp)
        if string[i] in specialchars or string[i] in variables:
            if not temp:
                proper = proper +string[i]
                continue
            if checkInt(temp):
                proper = proper + temp
            else:
                proper = proper + str(int(temp))
            proper = proper + string[i]
            temp = ""
        else:
            temp = temp + string[i]
        if i == len(string)-1:
            if not temp:
                continue
            proper = proper + str(int(temp))
    return proper
#Check EOL special char
def checkSpecialChars(string): 
    if not string[len(string)-1] in ["-", "+", "*", "/"]:
        temp2 = [m.start() for m in re.finditer("\*\*\*", string)]
        if len(temp2)!=0:
            return False
        temp2 = [m.start() for m in re.finditer("///", string)]
        if len(temp2)!=0:
            return False
        split3 = re.split("\)", string)
        for i in range(len(split3)):
            if i==0:
                continue
            temp = split3[i]
            if temp:
                if not temp[0] in ["-", "+", "*","/"]:
                    return False
        split1 = re.split("\*", string)
        emptyCount = 0
        for i in range(len(split1)):
            temp = split1[i]
            if not temp:
                emptyCount +=1
                continue
            if temp[len(temp)-1] in ["-", "+", "/"]:
                return False
        if emptyCount>1:
            return False
        emptyCount = 0
        split2 = re.split("/", string)
        for i in range(len(split2)):
            temp = split2[i]
            if not temp:
                emptyCount +=1
                continue
            if temp[len(temp)-1] in ["-", "+", "*"]:
                return False
    else:
        return False
    return True

m = main()
m.train()

cap = cv2.VideoCapture(0)
try:
    while(True):
        ret, frame = cap.read()
        vd = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5), np.uint8)
        #imopen = cv2.morphologyEx(vd, cv2.MORPH_OPEN, kernel)
        imblur = cv2.GaussianBlur(vd, (5,5), 0)
        ret,thresh = cv2.threshold(imblur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #_, thresh2 = cv2.threshold(thresh,127,255,cv2.THRESH_BINARY_INV)
        #ret,thresh = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        thresh1 = thresh
        image, con, hie = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #_, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
        imgList = []#skip blank frame in the last
        xcoordList = []
        ycoordList = []
        widthList = []
        heightList = []
        i=0;
        #previous x value and w value of the contour
        prevx = 0
        prevw = 0
        for c in con:
            if(cv2.contourArea(c)>50):    
                x, y, w, h = cv2.boundingRect(c)
                #if prevx+prevw > x:
                #    continue
                if(h>23 ):
                    imgrect = cv2.rectangle(image, (x-5,y-5), (x+w+5, y+h+5), (0,255,0),1)
                    digitImg = image[y:y+h, x:x+w]
                    
                    #resizing for prediction
                    if digitImg.shape[0]>0 and digitImg.shape[1]>0:
                        digit_resized = cv2.resize(digitImg, (45,45))
                        imgList.append(digit_resized)
                        xcoordList.append(x)
                        ycoordList.append(y)
                        widthList.append(w)
                        heightList.append(h)
            #else:
                else:
                    if(i+1<len(con)):
                        x1, y1, w1, h1 = cv2.boundingRect(c)
                        x2, y2, w2, h2 = cv2.boundingRect(con[i+1])
                        if(abs(x1-x2)<30):
                            imgrect = cv2.rectangle(image, (x2-5,y2-5), (x2+w2+5, y1+h1+5), (0,255,0),1)
                            digitImg = image[y2:y1+h1, x2:x2+w2]
                            
                            #resizing for prediction
                            if digitImg.shape[0]>0 and digitImg.shape[1]>0:
                                digit_resized = cv2.resize(digitImg, (45,45))
                                imgList.append(digit_resized)
                                xcoordList.append(x2)
                                ycoordList.append(y2)
                                widthList.append(w2)
                                heightList.append(y1+h1-y2)
                        else:
                            if(w1>3*h1):
                                imgrect = cv2.rectangle(image, (x1-5,y1-17), (x1+w1+5, y1+h1+17), (0,255,0),1)
                                digitImg = image[y1-15:y1+h1+15, x1:x1+w1]
                                
                                #resizing for prediction
                                if digitImg.shape[0]>0 and digitImg.shape[1]>0:
                                    digit_resized = cv2.resize(digitImg, (45,45))
                                    imgList.append(digit_resized)
                                    xcoordList.append(x1)
                                    ycoordList.append(y1)
                                    widthList.append(w1)
                                    heightList.append(h1)
                            #imgList.append(digitImg)
                #prevx = x
                #prevw = w
            i+=1
        
        #For prediction
        imgList = sort(imgList, xcoordList, ycoordList)
        imgArr = np.array(imgList)
        #TODO: add prediction over here for the imgArr
        imgPredict = m.test(imgArr.reshape(imgArr.shape[0], 45, 45, 1))
        imgPredict = np.array(imgPredict)
        
        expression = ""
        if len(imgPredict.shape)!=2:
            continue
        symInd = np.argmax(imgPredict, axis=1)
        for i in range(len(symInd)):
            #expression +=   m.ds.symbols[np.argmax(imgPredict, axis=1)]
            if symInd[i]>=len(symbols):
                continue
            expression +=   symbols[symInd[i]]
        #cv2.putText(imgrect, str(expression+"?"), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,153,255), 2, cv2.LINE_AA)
        if expression:
            varsIndices = None
            if eolCheck(expression):
                if checkSpecialChars(expression):
                    expression = invalidTokenCheck(expression)
                    varsIndices = [m.start() for m in re.finditer("m|p|w|y|b",expression)]
                    #operatorIndices = [m.start() for m in re.finditer("m|p|w|y|b",expression)]
                    
                    #print(expression)
                    try:
                        if len(varsIndices) == 1:
                            symb = Symbol(expression[varsIndices[0]])
                            expr = solveset(expression, symb)
                            cv2.putText(imgrect, str(expression), (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,153,255), 1, cv2.LINE_AA)
                            cv2.putText(imgrect, str(str(symb)+"="+str(expr)), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,153,255), 1, cv2.LINE_AA)
                        else:
                            if not varsIndices:
                                expr = sp.sympify(str(expression))
                                cv2.putText(imgrect, str(expression+"="+str(expr)), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,153,255), 2, cv2.LINE_AA)
                    except:
                        traceback.print_exc()
                        cv2.putText(imgrect, str(expression+"?"), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,153,255), 2, cv2.LINE_AA)
                        continue
                    #print(type(expr))
                    #print("**********")
                    
                else:
                    cv2.putText(imgrect, str(expression+"?"), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,153,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(imgrect, str(expression+"?"), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,153,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(imgrect, str(expression+"?"), (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,153,255), 2, cv2.LINE_AA)
        
        print(expression)
    
        cv2.namedWindow('beach',cv2.WINDOW_NORMAL)    
        cv2.imshow('beach',imgrect)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
        #time.sleep(0.1)
except Exception as e:
    traceback.print_exc()
    cap.release()
    cv2.destroyAllWindows()
    
    

cap.release()
cv2.destroyAllWindows()

