import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

SCREEN_WIDTH = int(cap.get(3))
SCREEN_HEIGHT = int(cap.get(4))

while True:
    ret, img = cap.read()
    img = img[:, ::-1, :]

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (90, 125, 50), (130, 255, 255))

    pixelsLeft = mask[0:SCREEN_HEIGHT, 0:SCREEN_WIDTH//2]
    pixelsLeftCount = np.sum(pixelsLeft == 255)

    pixelsRight = mask[0:SCREEN_HEIGHT, SCREEN_WIDTH // 2:]
    pixelsRightCount = np.sum(pixelsRight == 255)

    font = cv.FONT_HERSHEY_PLAIN
    mask = cv.line(mask, (SCREEN_WIDTH//2, 0), (SCREEN_WIDTH//2, SCREEN_HEIGHT), 255, 5)
    mask = cv.putText(mask, str(pixelsLeftCount), (0, 50), font, 4, (255, 255, 255), 5, cv.LINE_AA)
    mask = cv.putText(mask, str(pixelsRightCount), (SCREEN_WIDTH//2, 50), font, 4, (255, 255, 255), 5, cv.LINE_AA)

    cv.imshow('video', mask)
    cv.imshow('video2', img)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()