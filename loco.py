import cv2
from robotpy_apriltag import AprilTagDetector
from djitellopy import tello

detector = AprilTagDetector()
detector.addFamily("tag36h11")

tello = tello.Tello()
tello.connect()
print(tello.get_battery())

tello.streamoff()
tello.streamon()
frame_reader = tello.get_frame_read()

tello.takeoff()

tello.move_forward(82)

while True:
    frame = frame_reader.frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame",img)
    cv2.waitKey(1)
    img = cv2.rotate(img,cv2.ROTATE_180)
    img = cv2.flip(img,1)
    detections = detector.detect(img)
    for detection in detections:
        id = detection.getId()
        if id == 1:
            tello.flip_forward()
        elif id == 2:
            tello.flip_left()
        elif id == 3:
            tello.flip_right()
        elif id == 4:
            tello.land()
        else:
            tello.land()
            tello.end()