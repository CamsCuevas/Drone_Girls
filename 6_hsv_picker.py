import cv2
import numpy as np
import time
from djitellopy import Tello


# A required callback method that goes into the trackbar function.
def nothing(x):
    pass


# Initialize the Tello drone
tello = Tello()
tello.connect()

# Start the video stream
tello.streamon()

# Create a window named trackbars
cv2.namedWindow("Trackbars")

# Create trackbars to control HSV ranges
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    # Get the frame from the Tello camera
    frame = cv2.cvtColor(tello.get_frame_read().frame, cv2.COLOR_BGR2RGB)

    if frame is None:
        break

    # Flip the frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to HSV image
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Get the new values of the trackbars in real-time
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # Set the lower and upper HSV range
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    # Filter the image and get the binary mask
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Visualize the result by masking the frame
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the binary mask to a 3-channel image for stacking
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Stack the original frame, mask, and result side by side
    stacked = np.hstack((mask_3, frame, res))

    # Show the stacked images
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))

    # Check for key presses
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

    if key == ord('s'):  # 's' key to save HSV values
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(thearray)

        # Save the array as 'hsv_value.npy'
        np.save('hsv_value', thearray)
        break

# Release resources and close windows
tello.streamoff()
cv2.destroyAllWindows()