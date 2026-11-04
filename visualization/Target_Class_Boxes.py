### Draw ONLY target class boxes and save to output folder
import os
import cv2
import yaml

# ---------------------------
# Config
# ---------------------------
DATA_YAML = "../dataset/data.yaml"
IMAGE_FOLDER = "../dataset/images/train"
LABEL_FOLDER = "../dataset/labels/train"
OUTPUT_FOLDER = "../dataset/output_Neopycnodonte zibrowii"


#------------------------------------------------
# Select the target class name you want to filter
TARGET_CLASS_NAME = "Neopycnodonte zibrowii"
#------------------------------------------------


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------
# Load class names
# ---------------------------
with open(DATA_YAML, 'r') as f:
    data = yaml.safe_load(f)

CLASS_NAMES = [data["names"][i] for i in range(len(data["names"]))]

# 🔴 Find class ID for target
TARGET_CLASS_ID = None
for i, name in enumerate(CLASS_NAMES):
    if name == TARGET_CLASS_NAME:
        TARGET_CLASS_ID = i
        break

if TARGET_CLASS_ID is None:
    raise ValueError(f"Class '{TARGET_CLASS_NAME}' not found in data.yaml")

print(f"Target class '{TARGET_CLASS_NAME}' has ID: {TARGET_CLASS_ID}")

# ---------------------------
# Draw ONLY target class boxes
# ---------------------------
def draw_target_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return None

    height, width = img.shape[:2]

    found = False  # Track if target exists in image

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id, x_center, y_center, w, h = map(float, parts)

                # ✅ Only process target class
                if int(class_id) != TARGET_CLASS_ID:
                    continue

                found = True

                x_center *= width
                y_center *= height
                w *= width
                h *= height

                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Draw rectangle (fixed color for clarity)
                color = (0, 255, 0)  # green

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, TARGET_CLASS_NAME, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img if found else None  # Only return if target found


# ---------------------------
# Process all images
# ---------------------------
image_files = [f for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith(('.jpg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    label_path = os.path.join(LABEL_FOLDER, os.path.splitext(img_file)[0] + ".txt")

    img = draw_target_boxes(img_path, label_path)

    # ✅ Save only images that contain the target class
    if img is not None:
        output_path = os.path.join(OUTPUT_FOLDER, img_file)
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")

print("Done! Only target class boxes saved.")