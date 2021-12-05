import math

import numpy as np
import cv2
import time
from math import*

# bc = tan(45/radian)
# print(atan(1)*radian)#From value, calculate degree
objectName = 'STRAWBERRY'
cameraNo1 = 0
cameraNo2 = 2
cameraCenterX = 213
cameraCenterY = 120
cX = 0.0
cY = 0.0
X = 0.0
Y = 0.0
convertRadianToDegree = 180/math.pi
print(math.pi)
frameWidth = 450
frameHeight = 400

degreesOutput = 0.0
degreesOutputCam1 = 0.0
degreesOutputCam2 = 0.0
deepDistence = 0.0

set3 = 450
set4 = 400

captureFirstCamera = cv2.VideoCapture(cameraNo1, cv2.CAP_DSHOW)
captureFirstCamera.set(3, set3)
captureFirstCamera.set(4, set4)

captureSecondCamera = cv2.VideoCapture(cameraNo2, cv2.CAP_DSHOW)
captureSecondCamera.set(3, set3)
captureSecondCamera.set(4, set4)

def empty(a):
    pass

cv2.namedWindow("frameFirstCamera", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("frameFirstCamera", frameWidth, frameHeight)

cv2.namedWindow("frameSecondCamera", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("frameSecondCamera", frameWidth, frameHeight)

# cv2.createTrackbar("lowH", "frameFirstCamera", 0, 255, empty)
# cv2.createTrackbar("highH", "frameFirstCamera", 15, 255, empty)
# cv2.createTrackbar("lowS", "frameFirstCamera", 160, 255, empty)
# cv2.createTrackbar("highS", "frameFirstCamera", 255, 255, empty)
# cv2.createTrackbar("lowV", "frameFirstCamera", 168, 255, empty)
# cv2.createTrackbar("highV", "frameFirstCamera", 255, 255, empty)
#
# cv2.createTrackbar("lowH", "frameSecondCamera", 0, 255, empty)
# cv2.createTrackbar("highH", "frameSecondCamera", 15, 255, empty)
# cv2.createTrackbar("lowS", "frameSecondCamera", 160, 255, empty)
# cv2.createTrackbar("highS", "frameSecondCamera", 255, 255, empty)
# cv2.createTrackbar("lowV", "frameSecondCamera", 168, 255, empty)
# cv2.createTrackbar("highV", "frameSecondCamera", 255, 255, empty)

def calculateDeep(compareCam):
    global cX
    global cY
    global X
    global Y
    global degreesOutputCam1
    global degreesOutputCam2
    global deepDistence

    if compareCam == 1:
        print("cx", cX)
        print("cy", cY)
        T1D = abs(cameraCenterX - cX)
        anpha1 = atan(10.2 * T1D / 4451.7) * convertRadianToDegree
        degreesOutputCam1 = 90 - anpha1
        X = cX - cameraCenterX
        Y = cameraCenterY - cY
    else:
        T1D = abs(cameraCenterX - cX)
        beta1 = atan(10.2 * T1D / 4451.7) * convertRadianToDegree
        degreesOutputCam2 = 90 - beta1

    # print('degree1 =', degreesOutputCam1)
    # print('degree2 =', degreesOutputCam2)
    deepDistence = 8 * tan(degreesOutputCam1 / convertRadianToDegree) * tan(degreesOutputCam2 / convertRadianToDegree) / (tan(degreesOutputCam1 / convertRadianToDegree) + tan(degreesOutputCam2 / convertRadianToDegree))
    # print('distence =', deepDistence)

def processImage(frameNo, captureFrame, trackBar, frameShow, redMask, compareCam, scale=0.75):
    ret0, frameNo = captureFrame.read()
    global cX
    global cY
    global X
    global Y
    global deepDistence
    # width = int(captureFirstCamera.get(3))
    # height = int(captureFirstCamera.get(4))
    width = int(frameNo.shape[1] * scale)
    height = int(frameNo.shape[0] * scale)
    # print(height)
    # print(width)

    # cv2.putText(frameNo, "c", (216, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(frameNo, (cameraCenterX, cameraCenterY), 15, (0, 255, 0), 1)
    cv2.line(frameNo, (0, cameraCenterY), (width, 120), (0, 255, 0), 1)
    cv2.line(frameNo, (cameraCenterX, 0), (213, height), (0, 255, 0), 1)

    inputImageFilter = cv2.bilateralFilter(frameNo, 8, 8, 20)  # filter
    hsvImage = cv2.cvtColor(inputImageFilter, cv2.COLOR_BGR2HSV)  # convert color space

    # lowH = cv2.getTrackbarPos("lowH", trackBar)
    # highH = cv2.getTrackbarPos("highH", trackBar)
    # lowS = cv2.getTrackbarPos("lowS", trackBar)
    # highS = cv2.getTrackbarPos("highS", trackBar)
    # lowV = cv2.getTrackbarPos("lowV", trackBar)
    # highV = cv2.getTrackbarPos("highV", trackBar)
    # lowRed = np.array([lowH, lowS, lowV])
    # highRed = np.array([highH, highS, highV])

    lowRed = np.array([0, 158, 50])
    highRed = np.array([15, 255, 255])
    grayObject = cv2.inRange(hsvImage, lowRed, highRed)  # Detect object based on HSV Range values
    redObject = cv2.bitwise_and(frameNo, frameNo, mask=grayObject)

    contours, hierarchy = cv2.findContours(grayObject, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contours in contours:
        area = cv2.contourArea(contours)
        if area > 1000:
            cv2.drawContours(frameNo, contours, -1, (0, 255, 0), 2)
            M = cv2.moments(contours)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frameNo, (cX, cY), 0, (0, 0, 255), 0)
            cv2.putText(frameNo, "Center" + str(cX) + ", " + str(cY), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            calculateDeep(compareCam)
            cv2.putText(frameNo, "X: " + str(X), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frameNo, "Y: " + str(Y), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frameNo, "Z = D: " + str(deepDistence), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(frameShow, frameNo)
    cv2.imshow(redMask, redObject)

while True:
    processImage("frameFirstCamera", captureFirstCamera, "frameFirstCamera", "frameFirstCamera", "redMaskFirst", 1)
    processImage("frameSecondCamera", captureSecondCamera, "frameSecondCamera", "frameSecondCamera", "redMaskSecond", 2)

    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

captureFirstCamera.realese()
captureSecondCamera.realese()
cv2.destroyAllwindows()
