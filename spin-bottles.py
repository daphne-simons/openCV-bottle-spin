import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the PNG image with transparency
bottle_img = cv2.imread('heineken.png', cv2.IMREAD_UNCHANGED)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize variables to keep track of the previous frame and motion
prev_frame = None
paused = False
sensitivity = 20  # Initial sensitivity
cumulative_angles = []
motion_levels = []
face_positions = []

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
            continue

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Ensure the motion levels and angles lists are the same length as the number of faces detected
        while len(cumulative_angles) < len(faces):
            cumulative_angles.append(0)
            motion_levels.append(0)
            face_positions.append((0, 0))

        while len(cumulative_angles) > len(faces):
            cumulative_angles.pop()
            motion_levels.pop()
            face_positions.pop()

        for i, (x, y, w, h) in enumerate(faces):
            face_positions[i] = (x, y)

            # Calculate motion in the region of the detected face
            face_region = gray[y:y + h, x:x + w]
            prev_face_region = prev_frame[y:y + h, x:x + w]

            if prev_face_region.size == 0 or face_region.size == 0:
                continue

            motion_level, _ = calculate_motion(prev_face_region, face_region)
            motion_levels[i] = motion_level

            # Calculate the rotation speed for each face region
            speed = min(motion_level / 1000, 10)
            angle_increment = speed * 2  # Adjust this value to slow down the rotation

            # Update the cumulative angle
            cumulative_angles[i] = (cumulative_angles[i] + angle_increment) % 360

        prev_frame = gray

        # Rotate and overlay each bottle image for each detected face
        for i, (x, y) in enumerate(face_positions):
            rotated_bottle = rotate_image(bottle_img, cumulative_angles[i])

            # Get the position to overlay the image
            x_pos = x + w // 2 - rotated_bottle.shape[1] // 2
            y_pos = y + h // 2 - rotated_bottle.shape[0] // 2

            # Overlay the rotated image on the frame
            overlay_image_alpha(frame, rotated_bottle, x_pos, y_pos)

    # Show the frame with the overlay and speed text
    for i, (x, y) in enumerate(face_positions):
        cv2.putText(frame, f'Speed {i+1}: {motion_levels[i] / 1000:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Motion Sensitivity: {sensitivity:.2f}', (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
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
