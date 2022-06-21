# OPENCV Documentation
# https://docs.opencv.org/3.4/index.html

# ran into media pipe/protobuf bugs: https://discuss.streamlit.io/t/typeerror-descriptors-cannot-not-be-created
# -directly/25639/3


import cv2
import mediapipe as mp
import time
# Used to convert protobuf message to a dictionary.
from google.protobuf.json_format import MessageToDict


class HandDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # creating a hands object with an init of -> ( static_image_mode=False, max_num_hands=2, model_complexity=1,
        # min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # method to draw the points together

    def findHands(self, img, draw=True):
        # hands object only takes rgb
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # processes this img but we need to use its results
        # print(results.multi_hand_landmarks)  <-- test to see what it outputs

        # if it detects a hand, loop through each hand and draw the points for that hand and connect them
        if self.results.multi_hand_landmarks:  # if not None
            for handLMs in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, handLMs, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_num=0, draw=True):
        lmList = []  # list of landmark positions
        whichOne = ""
        if self.results.multi_hand_landmarks:  # if not None, it was made self.results so we could use in multiple funcs
            myHand = self.results.multi_hand_landmarks[hand_num]  # gets the first hand
            whichOne = MessageToDict(self.results.multi_handedness[0])['classification'][0]['label']        # left or right hand

            # loop gets the position of each point, coverts to pixels
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 12, (225, 0, 225), cv2.FILLED)

        return lmList, whichOne


def main():
    cap = cv2.VideoCapture(1)  # open camera for video capture, most devices should use 0 instead of 1

    pTime = 0  # previous and current time
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)) + "FPS", (10, 70), 1, 3, (225, 0, 225),
                    3)  # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
