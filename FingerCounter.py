# ------Resources------
# https://google.github.io/mediapipe/solutions/hands.html

# ------Imports--------
import cv2
import time
import os
import HandTrackingModule as htm

# -------Video Display-----------
wCam, hCam = 640, 480  # dimensions of our capture window
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

# ------Images of Finger Count----------
folderPath = "images"
my_list = os.listdir(folderPath)
# print(my_list)
overlay_list = []
for imPath in my_list:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlay_list.append(image)


# ------Look at each
def openOrClosedCount(landmarksList, whichHand):
    # right hand only at the moment
    fingers = []
    tipIds = [4, 8, 12, 16, 20]

    if whichHand == "Left":  # check thumb is closed, thumb is a special case, checking the x-coor instead
        if landmarksList[tipIds[0]][1] > landmarksList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    elif whichHand == "Right":
        # check thumb is closed, thumb is a special case, checking the x-coor instead
        if landmarksList[tipIds[0]][1] < landmarksList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # checking if the 4 fingers are closed
    for id in range(1, 5):

        if landmarksList[tipIds[id]][2] < landmarksList[tipIds[id] - 2][2]:  # if y-coor of finger tip < y-coor base of same finger, then finger closed
            fingers.append(1)
        else:
            fingers.append(0)
    # print(fingers)

    total = fingers.count(1)
    return total


# --------Main Program----------
pTime = 0
cTime = 0
totalFingers = 0
detector = htm.HandDetector(detectionCon=0.75)
while True:
    success, img = cap.read()
    img = detector.findHands(img)

    # flip image
    img = cv2.flip(img, 1)

    lmList, label = detector.findPosition(img, draw=False)

    # print(lmList)
    if len(lmList) != 0:
        totalFingers = openOrClosedCount(lmList, label)

        # we slice to make the size of our image display, our images are  200x200
        h, w, c = overlay_list[totalFingers - 1].shape
        img[0:h, 0:w] = overlay_list[totalFingers - 1]  # zero is taken care of because we access image list from back w/ -1

        #     img,  start point, end points, colour, fill or no?
        cv2.rectangle(img, (20, 200), (170, 425), (255, 231, 135), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), 1, 10, (124, 108, 119), 25)

    # setting our fps variables
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)) + "FPS", (490, 50), 1, 3, (124, 108, 119,), 3)  # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

    # because we are flipping the image, the label gets a little wonky, it needs to be the opposite
    if label == "Right":
        opp_label = "Left"
    elif label == "Left":
        opp_label = "Right"
    else:
        opp_label = ""

    cv2.putText(img, opp_label, (500, 90), 1, 3, (124, 108, 119), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
