import cv2 as cv
import HandTrackingModule as hd

cap = cv.VideoCapture(1)
detector = hd.handDetector()
while True:
  sucess, img = cap.read()
  img = detector.findHands(img)
  try:
    fingers = detector.getFingers(img)
    print(fingers)
  except Exception as ex:
    print(f'An Exception Occurred: {ex}')
  cv.imshow('image', img)
  k = cv.waitKey(1)
  if k == 27:
    break