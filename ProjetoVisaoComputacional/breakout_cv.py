import cv2 as cv2
import numpy as np
import os

def cv_setup(game):
    dados = Dados
    cv_init(game, dados)
    cv_update(game, dados)


class Dados:
    def __init__(self, img, track_window, term_critic, roi_hist):
        self.img = img
        self.track_window = track_window
        self.term_critic = term_critic
        self.hsv_roi = roi_hist





def cv_init(game, dados):

    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)
    # rest of init
    ret, frame1 = game.cap.read()
    # setup initial location of window
    x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
    dados.img = cv2.rectangle(frame1, (x, y), (x + w, y + h), 255, 2)

    dados.track_window = (x, y, w, h)

    # set up the ROI for tracking
    roi = frame1[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (90, 125, 50), (130, 255, 255))
    dados.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(dados.roi_hist, dados.roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    dados.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)




def cv_update(game,dados):
    cap = game.cap
    if not cap.isOpened():
        cap.open(-1)

    SCREEN_WIDTH = int(cap.get(3))
    SCREEN_HEIGHT = int(cap.get(4))

    ret, image = cap.read()
    image = image[:, ::-1, :]

    fgmask = cv_process(cap, SCREEN_WIDTH, SCREEN_HEIGHT, game, dados)
    cv_output(image, fgmask)
    #game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(cap, SWidth, SHeight, game, dados):
    # main image processing code
    ret, image = cap.read()
    image = image[:, ::-1, :]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 125, 50), (130, 255, 255))



    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], dados.roi_hist, [0, 180], 1)

    # apply camshift to get the new location
    ret, track_window = cv2.CamShift(dst, dados.track_window, dados.term_crit)

    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(image, [pts], True, 255, 2)

    return img2


def cv_output(image, fgmask):
    cv2.imshow("Image", image)
    cv2.imshow("MaskedImage", fgmask)
    # rest of output rendering
    cv2.waitKey(1)
