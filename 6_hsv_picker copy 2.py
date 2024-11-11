import cv2
import numpy as np

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()  
    cv2.imshow("Tello Camera", frame)
    cv2.waitKey(1)

    if frame is None:
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

    cv2.imshow("filtered",res)
    cv2.waitKey(1)

    key = cv2.waitKey(1)

    if key == ord('s'):  
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(thearray)

        np.save('hsv_value', thearray)
        break

cv2.destroyAllWindows()