import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(1)

# Load the PNG image with transparency
bottle_img = cv2.imread('heineken.png', cv2.IMREAD_UNCHANGED)  # Make sure the path to your PNG file is correct

# Initialize variables to keep track of the previous frame and motion
prev_frame = None
motion_level = 0
paused = False

def calculate_motion(frame1, frame2):
    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    # Threshold the image to binarize it
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
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
            continue

        # Calculate the motion level and contours
        motion_level, contours = calculate_motion(prev_frame, gray)
        prev_frame = gray

        # Draw rectangles around detected motion
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust the contour area threshold as needed
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the rotation speed
        speed = min(motion_level / 1000, 10)  # Adjust the speed factor as needed
        angle = speed * 10  # Example: the angle can be 10 times the speed

        # Rotate the PNG image
        rotated_bottle = rotate_image(bottle_img, angle)

        # Get the position to overlay the image (e.g., center of the frame)
        x_pos = (frame.shape[1] - rotated_bottle.shape[1]) // 2
        y_pos = (frame.shape[0] - rotated_bottle.shape[0]) // 2

        # Overlay the rotated image on the frame
        overlay_image_alpha(frame, rotated_bottle, x_pos, y_pos)

    # Show the frame with the overlay and speed text
    cv2.putText(frame, f'Bottle Spinning Speed: {speed:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Motion-Based Spinning Bottle', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
