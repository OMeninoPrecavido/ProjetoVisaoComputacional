import cv2 as cv2
import numpy as np


def cv_setup(game):
    cv_init(game)
    cv_update(game)


def cv_init(game):
    global roi_hist, track_window, term_crit

    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)

    # rest of init

    ret, frame = game.cap.read()

    # setup initial location of window
    x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
    track_window = (x, y, w, h)

    # set up the ROI for tracking
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((115., 150., 32.)), np.array((179., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Set up the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

def cv_update(game):
    cap = game.cap
    if not cap.isOpened():
        cap.open(-1)
    ret, image = cap.read()
    cv_process(image, ret)

    if track_window[0] > game.cap.get(3)//2:
        game.paddle.move(-10)
    else:
        game.paddle.move(10)
    game.after(1, cv_update, game)


def cv_process(frame, ret):
    global track_window

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img2', img2[:, ::-1, :])
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            return
    else:
        return
    pass


def cv_output(image):
    cv2.imshow("Image", image[:, ::-1, :])
    # rest of output rendering
    cv2.waitKey(1)
