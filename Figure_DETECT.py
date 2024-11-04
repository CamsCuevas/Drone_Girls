import cv2
from djitellopy import Tello
import numpy as np

# Inicializar y comenzar el stream de video Tello
tello = Tello()
tello.connect()
tello.streamon()

# Iniciar el loop de procesamiento de video
while True:
    # Obtener frame de la cámara del Tello
    frame = tello.get_frame_read().frame

    # Redimensionar el frame para procesar más rápido
    frame = cv2.resize(frame, (640, 480))

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Colores de interes cambiar de acuerdo a hsv_picker tello
    lower_black = np.array([26, 171, 76])       # Minimum value for black
    upper_black = np.array([180, 255, 255])     # Maximum value for black

    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # Aplicar un umbral para binarizar la imagen
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Aproximar el contorno para detectar la forma
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Dibujar el contorno en la imagen original
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

        # Identificar la forma basada en el número de lados
        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            shape_name = "Rectangle"  # o cuadrado
        elif len(approx) == 5:
            shape_name = "Pentagon"
        elif len(approx) == 10:
            shape_name = "Star"
        else:
            shape_name = "Circle"

        # Obtener coordenadas del texto
        x, y = approx[0][0]

        # Poner el nombre de la forma en la imagen
        cv2.putText(frame, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Mostrar la imagen procesada
    cv2.imshow("Tello Camera", frame)

    # Romper el loop con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizar
tello.streamoff()
cv2.destroyAllWindows()