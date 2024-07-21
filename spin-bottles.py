import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the PNG image with transparency
bottle_img = cv2.imread('heineken.png', cv2.IMREAD_UNCHANGED)
duplicate_img = bottle_img.copy()  # Duplicate the original image

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables to keep track of the previous frames and motion
prev_frame = None
prev_frame_left = None
prev_frame_right = None
motion_level_left = 0
motion_level_right = 0
paused = False
sensitivity = 20  # Initial sensitivity
cumulative_angle_left = 0  # Initialize the cumulative angle for the left image
cumulative_angle_right = 0  # Initialize the cumulative angle for the right image

def calculate_motion(frame1, frame2):
    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    # Threshold the image to binarize it
    _, thresh = cv2.threshold(blur, sensitivity, 255, cv2.THRESH_BINARY)
    # Find the contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate the total area of the contours
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    return total_area, contours

def rotate_image(image, angle):
    # Get the image size
    h, w = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)
    # Create a rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def overlay_image_alpha(img, img_overlay, x, y):
    """ Overlay `img_overlay` onto `img` at position (x, y) and blend using alpha channel. """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if no overlap
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_alpha = img_overlay[y1o:y2o, x1o:x2o, 3] / 255.0
    img_alpha_inv = 1.0 - img_alpha

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (img_alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                img_alpha_inv * img[y1:y2, x1:x2, c])

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    if not paused:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            prev_frame_left = gray[:, :gray.shape[1] // 2]
            prev_frame_right = gray[:, gray.shape[1] // 2:]
            continue

        # Split the frame into left and right halves
        gray_left = gray[:, :gray.shape[1] // 2]
        gray_right = gray[:, gray.shape[1] // 2:]

        # Calculate the motion level and contours for each half
        motion_level_left, contours_left = calculate_motion(prev_frame_left, gray_left)
        motion_level_right, contours_right = calculate_motion(prev_frame_right, gray_right)

        prev_frame_left = gray_left
        prev_frame_right = gray_right

        # Calculate the rotation speed for each image
        speed_left = min(motion_level_left / 1000, 10)  # Adjust the speed factor as needed
        speed_right = min(motion_level_right / 1000, 10)  # Adjust the speed factor as needed

        angle_increment_left = speed_left * 2  # Adjust this value to slow down the rotation
        angle_increment_right = speed_right * 2  # Adjust this value to slow down the rotation

        # Update the cumulative angles
        cumulative_angle_left = (cumulative_angle_left + angle_increment_left) % 360
        cumulative_angle_right = (cumulative_angle_right + angle_increment_right) % 360

        # Rotate the PNG images
        rotated_bottle_left = rotate_image(bottle_img, cumulative_angle_left)
        rotated_bottle_right = rotate_image(duplicate_img, cumulative_angle_right)

        # Get the positions to overlay the images
        x_pos_left = (frame.shape[1] // 4) - (rotated_bottle_left.shape[1] // 2)
        y_pos_left = (frame.shape[0] - rotated_bottle_left.shape[0]) // 2

        x_pos_right = (3 * frame.shape[1] // 4) - (rotated_bottle_right.shape[1] // 2)
        y_pos_right = (frame.shape[0] - rotated_bottle_right.shape[0]) // 2

        # Overlay the rotated images on the frame
        overlay_image_alpha(frame, rotated_bottle_left, x_pos_left, y_pos_left)
        overlay_image_alpha(frame, rotated_bottle_right, x_pos_right, y_pos_right)

    # Show the frame with the overlay, speed text, and sensitivity text
    cv2.putText(frame, f'Bottle Spinning Speed Left: {speed_left:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Bottle Spinning Speed Right: {speed_right:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Motion Sensitivity: {sensitivity:.2f}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Motion-Based Spinning Bottle', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('+'):
        sensitivity = min(sensitivity + 5, 100)
    elif key == ord('-'):
        sensitivity = max(sensitivity - 5, 0)

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
