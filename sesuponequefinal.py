import cv2
import numpy as np
from figuras import is_star, is_circle, is_square, is_pentagon, is_triangle

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()  
    if not ret:
        print("Error al capturar el frame")
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    l_h = 95
    l_s = 108
    l_v = 143
    u_h = 106
    u_s = 255
    u_v = 255

    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cnts, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_minima=700
    contornos_filtrados = []
    for contorno in cnts:
        area = cv2.contourArea(contorno)
        if area >= area_minima:
            contornos_filtrados.append(contorno)

    figura = None

    for c in contornos_filtrados:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if is_triangle(approx):
            cv2.putText(res, 'Triangulo', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            figura = "triangulo"

        elif is_square(approx):
            cv2.putText(res, 'Cuadrado', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            figura = "cuadrado"

        elif is_pentagon(approx):
            cv2.putText(res, 'Pentagono', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            figura = "pentagono"

        elif is_star(approx):
            cv2.putText(res, 'Estrella', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            figura = "estrella"

        elif is_circle(approx):
            cv2.putText(res, 'Circulo', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            figura = "circulo"

        if figura:
            cv2.drawContours(res, [approx], -1, (0, 255, 0), 2)

    cv2.imshow("filtered",res)

    key = cv2.waitKey(1)

    if key == ord('s'):  
        break

cap.release()
cv2.destroyAllWindows()