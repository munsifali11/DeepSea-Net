### Script to run YOLOv8 detection on a folder of images and save the detected images to an output folder.
from ultralytics import YOLO
import os

# Model path
model_path = "../weights/best.pt"

# Folder containing test or validation images
image_folder = "../images/val"

# Output folder for detected images
output_folder = "../outputs/val_predictions"

# Load YOLO model
model = YOLO(model_path)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Run detection on all images in the folder
results = model.predict(
    source=image_folder,
    conf=0.5,
    save=True,
    save_txt=False,
    project=output_folder,
    name="detected_images",
    exist_ok=True
)

print(f"Done! Detected images saved in: {os.path.join(output_folder, 'detected_images')}")