from djitellopy import Tello
import cv2
import numpy as np

# Connect to Tello
tello = Tello()
tello.connect()
tello.streamon()


def draw_dashed_line(img, x, color=(0, 255, 0), thickness=2, dash_length=10, gap_length=10):
    """Draws a vertical dashed line at the specified x coordinate."""
    height = img.shape[0]
    for y in range(0, height, dash_length + gap_length):
        cv2.line(img, (x, y), (x, y + dash_length), color, thickness)


def process_frame(frame):
    # Convert to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dir = ""

    # Define range for black color in HSV (USE 6_hsv_picker_tello.py to adjust the color of interest)
    lower_black = np.array([0, 0, 0])  # Minimum value for black
    upper_black = np.array([180, 255, 50])  # Maximum value for black

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Optionally, apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Convert the mask to a 3-channel image to allow colored drawing
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Get the center x-coordinate for the vertical dashed line
    center_x = mask_colored.shape[1] // 2

    # Draw a vertical dashed line at the center of the image (in white)
    draw_dashed_line(mask_colored, center_x, color=(255, 255, 255), thickness=2, dash_length=10, gap_length=10)

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming the largest contour is the line
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the width of the contour's bounding box to verify it's the correct line
        x, y, w, h = cv2.boundingRect(largest_contour)
        if 7 <= w / 10 <= 9:  # Ensure the width is approximately 8 cm (scaled to the camera's field of view)

            # Draw the contour in green for visualization (on the colored mask)
            cv2.drawContours(mask_colored, [largest_contour], -1, (0, 255, 0), 2)  # Green contour

            # Get the moments of the contour to find the center
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])  # Center X of the contour
                cy = int(M["m01"] / M["m00"])  # Center Y of the contour

                # Draw the center point on the mask in red
                cv2.circle(mask_colored, (cx, cy), 5, (0, 0, 255), -1)  # Red circle for the center point

                # Use the center of the line to calculate deviation and send commands
                height, width, _ = mask_colored.shape
                deviation = cx - width // 2

                # Control logic to adjust Tello's movement based on deviation
                if deviation < -50:
                    # tello.move_left(20)
                    dir = "left"  # ←
                elif deviation > 50:
                    # tello.move_right(20)
                    dir = "right"  # →
                else:
                    # tello.move_forward(20)
                    dir = "foward"  # ↑

    return mask_colored, dir


while True:
    # Get frame from Tello
    frame = tello.get_frame_read().frame

    # Process frame to get the mask and track the black line
    mask_colored, dir = process_frame(frame)

    # Show the binarized mask with green contour, red center point, and dashed vertical line
    cv2.putText(mask_colored, dir, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Tello Line Tracking Mask", mask_colored)
    print(dir)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
tello.end()