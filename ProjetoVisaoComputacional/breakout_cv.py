import cv2 as cv2
import numpy as np

def cv_setup(game):
    cv_init(game)
    cv_update(game)


def cv_init(game):
    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)
    # rest of init


def cv_update(game):
    cap = game.cap
    if not cap.isOpened():
        cap.open(-1)

    SCREEN_WIDTH = int(cap.get(3))
    SCREEN_HEIGHT = int(cap.get(4))

    ret, image = cap.read()
    image = image[:, ::-1, :]
    mask = cv_process(image, SCREEN_WIDTH, SCREEN_HEIGHT, game)
    cv_output(image, mask)
    #game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(image, SWidth, SHeight, game):
    # main image processing code

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 125, 50), (130, 255, 255))

    pixelsLeft = mask[0:SHeight, 0:SWidth // 2]
    pixelsLeftCount = np.sum(pixelsLeft == 255)

    pixelsRight = mask[0:SHeight, SWidth // 2:]
    pixelsRightCount = np.sum(pixelsRight == 255)

    font = cv2.FONT_HERSHEY_PLAIN
    mask = cv2.line(mask, (SWidth // 2, 0), (SWidth // 2, SHeight), 255, 5)
    mask = cv2.putText(mask, str(pixelsLeftCount), (0, 50), font, 4, (255, 255, 255), 5, cv2.LINE_AA)
    mask = cv2.putText(mask, str(pixelsRightCount), (SWidth // 2, 50), font, 4, (255, 255, 255), 5, cv2.LINE_AA)

    if pixelsLeftCount > pixelsRightCount:
        game.paddle.move(-5)
    else:
        game.paddle.move(5)

    return mask


def cv_output(image, mask):
    cv2.imshow("Image", image)
    cv2.imshow("MaskedImage", mask)
    # rest of output rendering
    cv2.waitKey(1)
