import cv2 as cv2

def cv_setup(game):
    cv_init(game)
    cv_update(game)


def cv_init(game):
    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)
    # rest of init

    global RecX # posição x do retângulo
    RecX = game.cap.get(3)//2

    global face_cascade # haar cascade de face
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


def cv_update(game):
    cap = game.cap
    if not cap.isOpened():
        cap.open(-1)
    ret, image = cap.read()
    cv_process(image)
    cv_output(image)

    if RecX < (cap.get(3)//2):
        game.paddle.move(5)
    elif RecX > (cap.get(3)//2):
        game.paddle.move(-5)

    game.after(1, cv_update, game)


def cv_process(image):
    # main image processing code

    global RecX

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # retorna a posição de todos os rostos
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
        RecX = x

    pass


def cv_output(image):
    cv2.imshow("Image", image[:, ::-1, :])
    # rest of output rendering
    cv2.waitKey(1)
