# ---------------------------
# Draw YOLO bounding boxes and save to output folder
# ---------------------------
import os
import cv2
import yaml
# ---------------------------
# Config
# ---------------------------
DATA_YAML = "../dataset/data.yaml"  # Path to your data.yaml
IMAGE_FOLDER = "../dataset/images/train"
LABEL_FOLDER = "../dataset/labels/train"
OUTPUT_FOLDER = "../dataset/output"

# Select how many images to process (set to None to process all)
NUM_IMAGES = 4700  # How many images to process

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ---------------------------
# Load class names from data.yaml
# ---------------------------
with open(DATA_YAML, 'r') as f:
    data = yaml.safe_load(f)
CLASS_NAMES = data.get("names", {})  # dictionary {id: name}
# Convert to list ordered by class ID
CLASS_NAMES = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]

import random
# ---------------------------
# Generate a color for each class
# ---------------------------
CLASS_COLORS = {}
for i, class_name in enumerate(CLASS_NAMES):
    # Random bright colors
    CLASS_COLORS[i] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

# ---------------------------
# Function to draw YOLO boxes
# ---------------------------
def draw_yolo_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)

                x_center *= width
                y_center *= height
                w *= width
                h *= height
                x1 = int(x_center - w/2)
                y1 = int(y_center - h/2)
                x2 = int(x_center + w/2)
                y2 = int(y_center + h/2)

                class_name = CLASS_NAMES[int(class_id)]
                color = CLASS_COLORS[int(class_id)]  # use unique color per class

                # Draw rectangle and class name
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, class_name, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

# ---------------------------
# Process first NUM_IMAGES images
# ---------------------------
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png'))][:NUM_IMAGES]

for img_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    label_path = os.path.join(LABEL_FOLDER, os.path.splitext(img_file)[0] + ".txt")

    img = draw_yolo_boxes(img_path, label_path)
    if img is not None:
        output_path = os.path.join(OUTPUT_FOLDER, img_file)
        # Convert RGB back to BGR for cv2.imwrite
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

print("Done! Bounding box images saved in:", OUTPUT_FOLDER)
