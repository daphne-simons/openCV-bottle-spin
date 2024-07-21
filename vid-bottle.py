import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the video file
video_path = 'vid2.mp4'  # Replace with your video file path
video_cap = cv2.VideoCapture(video_path)
if not video_cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize variables to keep track of the previous frame and motion
prev_frame = None
paused = False
sensitivity = 20  # Initial sensitivity
fps = video_cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Initial frame delay in milliseconds

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
    return total_area

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

        # Calculate the motion level
        motion_level = calculate_motion(prev_frame, gray)
        prev_frame = gray

        # Adjust the playback speed of the video based on motion level
        speed_factor = max(min(motion_level / 1000, 10), 0.1)  # Avoid division by zero and too low speed
        new_delay = int(frame_delay / speed_factor)

        # Read the next frame from the video
        ret, video_frame = video_cap.read()
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            ret, video_frame = video_cap.read()
            if not ret:
                break

        # Add the text to the video frame
        cv2.putText(video_frame, f'Motion Level: {motion_level:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(video_frame, f'Motion Sensitivity: {sensitivity:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the video frame
        cv2.imshow('Motion-Controlled Video Playback', video_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF  # Use a short delay to control responsiveness
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('+'):
        sensitivity = min(sensitivity + 5, 100)
    elif key == ord('-'):
        sensitivity = max(sensitivity - 5, 0)

    # Wait based on the new delay
    cv2.waitKey(new_delay)

# Release the resources
cap.release()
video_cap.release()
cv2.destroyAllWindows()
