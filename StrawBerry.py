import cv2
import math
from math import*
import numpy as np
import tensorflow as tf
from tensorflow import keras

X = 0.0
Y = 0.0
cX = 0.0
cY = 0.0
index = 0
cameraNo1 = 0
cameraNo2 = 2
cameraCenterX = 213
cameraCenterY = 120

check1 = 0 # check if the object is in the common area of 2 cams
check2 = 0 # check if the object is in the common area of 2 cams
numberObjectInOneShot = 0

# lengthOfOnePixelX = 9.8/426
# lengthOfOnePixelY = 5.45/240
frameWidth = 450
frameHeight = 400
deepDistence = 0.0
distenceBetweenTwoCam = 8

beta1 = 0.0
anpha1 = 0.0
anpha2 = 0.0
degreesOutput = 0.0
degreesOutputCam1 = 0.0
degreesOutputCam2 = 0.0
convertRadianToDegree = 180/math.pi

readD0FromPlc = 0
activeImageProcess = 1

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

new_model = tf.keras.models.load_model('C:\HTCDT_PC_INTERFACE_code\AiFile\Xception.h5')
new_model.summary()
def writeToPlc():
    global cX
    global cY
    global deepDistence
    sendX = 0
    sendY = 0
    sendDeepDistence = 0

def aiProcess():
    global new_model
    if index >= 1:
        image_size = (180, 180)
        img = keras.preprocessing.image.load_img('C:\Image\image' + str(index) + '.png', target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = new_model.predict(img_array)
        score = predictions[0]
        ripePercent = 100 * (1 - score)
        print("ripe percent = ", ripePercent)
        if ripePercent >= 80:
            print("good")
            writeToPlc()

def calculateXYZ(compareCam):
    global cX
    global cY
    global X
    global Y
    global degreesOutputCam1
    global degreesOutputCam2
    global anpha1
    global anpha2
    global beta1
    global deepDistence
    global check1
    global check2

    if compareCam == 1:
        check1 += 1
        T1D = abs(cameraCenterX - cX)
        T2D = abs(cameraCenterY - cY)
        anpha1 = atan(9.8 * T1D / 4279.8) * convertRadianToDegree
        anpha2 = atan(5.45 * T2D / 2352) * convertRadianToDegree
        degreesOutputCam1 = 90 - anpha1
    else:
        check2 += 1
        T1D = abs(cameraCenterX - cX)
        beta1 = atan(9.8 * T1D / 4279.8) * convertRadianToDegree
        degreesOutputCam2 = 90 - beta1

    if check1 >= 1 & check2 >= 1:
        check1 = 0
        check2 = 0
        deepDistence = distenceBetweenTwoCam * tan(degreesOutputCam1 / convertRadianToDegree) * tan(degreesOutputCam2 / convertRadianToDegree) / (tan(degreesOutputCam1 / convertRadianToDegree) + tan(degreesOutputCam2 / convertRadianToDegree))
        if cX > cameraCenterX:
            X = tan(anpha1/convertRadianToDegree) * deepDistence
        else:
            X = -tan(anpha1 / convertRadianToDegree) * deepDistence
        if cY > cameraCenterY:
            Y = -tan(anpha2 / convertRadianToDegree) * deepDistence
        else:
            Y = tan(anpha2 / convertRadianToDegree) * deepDistence
        print("X: ", X)
        print("Y: ", Y)

def cropImage(inputImage):
    global cX
    global cY
    global index
    xLeft = cX - 45
    xRight = cX + 45
    yBottom = cY + 40
    yTop = cY - 40

    if xLeft > 0 and xRight > 0 and yBottom > 0 and yTop > 0:
        cv2.rectangle(inputImage, (cX - 40, cY + 40), (cX + 40, cY - 40), (0, 255, 0), 3)
        roi = inputImage[yTop: yBottom, xLeft: xRight] #Y && X
        index += 1
        cv2.imwrite('C:\Image\image' + str(index) + '.png', roi)
        cv2.imshow("roi", roi)

def processImage(frameNo, captureFrame, trackBar, frameShow, redMask, compareCam, scale=0.75):
    ret0, frameNo = captureFrame.read()
    global cX
    global cY
    global X
    global Y
    global deepDistence
    global cameraCenterX
    global numberObjectInOneShot
    width = int(frameNo.shape[1] * scale)
    height = int(frameNo.shape[0] * scale)

    # cv2.putText(frameNo, "c", (216, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.circle(frameNo, (cameraCenterX, cameraCenterY), 15, (0, 255, 0), 1)
    # cv2.line(frameNo, (0, cameraCenterY), (width, 120), (0, 255, 0), 1)
    # cv2.line(frameNo, (cameraCenterX, 0), (213, height), (0, 255, 0), 1)

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
            cv2.circle(frameNo, (cX, cY), 3, (0, 0, 255), -1)
            cv2.putText(frameNo, "Center" + str(cX) + ", " + str(cY), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            calculateXYZ(compareCam)
            if deepDistence > 0: # crop image just when the object is lies on common area two cams
                cropImage(frameNo)
                # numberObjectInOneShot += 1

            if compareCam == 2: # show X,Y,Z immediately after the calculation is complete
                cv2.putText(frameNo, "X: " + str(round(X, 2)), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frameNo, "Y: " + str(round(Y, 2)), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frameNo, "Z = D: " + str(round(deepDistence, 2)), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow(frameShow, frameNo)
    # cv2.imshow(redMask, redObject)

while True:
    activeAiProcess = 0
    # read D0 from PLC,
    # if D0 = 1:
        # activeImageProcess = 1
        # Write D0 = 0 to PLC

    if activeImageProcess == 1:
        processImage("frameFirstCamera", captureFirstCamera, "frameFirstCamera", "frameFirstCamera", "redMaskFirst", 1)
        processImage("frameSecondCamera", captureSecondCamera, "frameSecondCamera", "frameSecondCamera", "redMaskSecond", 2)

        while index > 0:
            activeAiProcess += 1
            if index >= activeAiProcess:
                aiProcess()
            else:
                index = 0
                activeAiProcess = 0

    X = 0
    Y = 0
    deepDistence = 0
    # activeImageProcess == 0

    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

# captureFirstCamera.realese()
# captureSecondCamera.realese()
# cv2.destroyAllwindows()
