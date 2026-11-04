import cv2
from ultralytics import YOLO
import numpy as np

# Load the trained YOLO model
model = YOLO('..\weights\best.pt')

# Open the video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video info (frame width, height, fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare the output video file
output_path = 'output_video_40_final.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process each frame from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model prediction on the frame
    results = model(frame)  # Predict the objects in the frame

    if results:
        if isinstance(results, list):
            for result in results:
                frame = result.plot()  # Draw bounding boxes on the frame
        else:
            frame = results.plot()  # Draw bounding boxes on the frame

    # Write the frame to the output video
    out.write(frame)
    cv2.imshow("Video with YOLO Predictions", frame)
    # Wait for 1 ms and check if user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything after processing is complete
cap.release()
out.release()
cv2.destroyAllWindows()