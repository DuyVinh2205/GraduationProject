import numpy as np
import cv2
import time

objectName = 'STRAWBERRY'
# cameraName = ''
cameraNo1 = 0
cameraNo2 = 2
frameWidth = 400
frameHeight = 450

set3 = 400
set4 = 450

captureFirstCamera = cv2.VideoCapture(cameraNo1, cv2.CAP_DSHOW)
captureFirstCamera.set(3, set3)
captureFirstCamera.set(4, set4)

captureSecondCamera = cv2.VideoCapture(cameraNo2, cv2.CAP_DSHOW)
captureSecondCamera.set(3, set3)
captureSecondCamera.set(4, set4)

def empty(a):
    pass

cv2.namedWindow("frameFirstCamera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frameFirstCamera", frameWidth, frameHeight)

cv2.namedWindow("frameSecondCamera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frameSecondCamera", frameWidth, frameHeight)

cv2.createTrackbar("lowH", "frameFirstCamera", 0, 255, empty)
cv2.createTrackbar("highH", "frameFirstCamera", 15, 255, empty)
cv2.createTrackbar("lowS", "frameFirstCamera", 160, 255, empty)
cv2.createTrackbar("highS", "frameFirstCamera", 255, 255, empty)
cv2.createTrackbar("lowV", "frameFirstCamera", 168, 255, empty)
cv2.createTrackbar("highV", "frameFirstCamera", 255, 255, empty)

cv2.createTrackbar("lowH", "frameSecondCamera", 0, 255, empty)
cv2.createTrackbar("highH", "frameSecondCamera", 15, 255, empty)
cv2.createTrackbar("lowS", "frameSecondCamera", 160, 255, empty)
cv2.createTrackbar("highS", "frameSecondCamera", 255, 255, empty)
cv2.createTrackbar("lowV", "frameSecondCamera", 168, 255, empty)
cv2.createTrackbar("highV", "frameSecondCamera", 255, 255, empty)

def processImage(frameNo, captureFrame, trackBar, frameShow, redMask):
    ret0, frameNo = captureFrame.read()
    inputImageFilter = cv2.bilateralFilter(frameNo, 8, 8, 20)  # filter
    hsvImage = cv2.cvtColor(inputImageFilter, cv2.COLOR_BGR2HSV)  # convert color space

    lowH = cv2.getTrackbarPos("lowH", trackBar)
    highH = cv2.getTrackbarPos("highH", trackBar)
    lowS = cv2.getTrackbarPos("lowS", trackBar)
    highS = cv2.getTrackbarPos("highS", trackBar)
    lowV = cv2.getTrackbarPos("lowV", trackBar)
    highV = cv2.getTrackbarPos("highV", trackBar)
    lowRed = np.array([lowH, lowS, lowV])
    highRed = np.array([highH, highS, highV])

    # lowRed = np.array([0, 158, 50])
    # highRed = np.array([15, 255, 255])
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
            cv2.putText(frameNo, "Center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(frameShow, frameNo)
    cv2.imshow(redMask, redObject)

while True:
    processImage('frameFirstCamera', captureFirstCamera,"frameFirstCamera", 'frameFirstCamera', 'redMaskFirst')
    processImage('frameSecondCamera', captureSecondCamera, "frameSecondCamera", 'frameSecondCamera', 'redMaskSecond')

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

captureFirstCamera.realese()
captureSecondCamera.realese()
cv2.destroyAllwindows()
