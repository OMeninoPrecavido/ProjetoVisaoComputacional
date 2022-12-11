import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # retorna a posição de todos os rostos
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

    cv2.imshow('frame', frame[:, ::-1, :])
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 2 problemas:
# 1 - onde deixar o face cascade? ✔
# 2 - onde como pegar a posição do retângulo ✔
#     - criar uma variavel e igualar ela ao x do retangulo durante a exec do for