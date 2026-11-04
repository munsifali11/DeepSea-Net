
from ultralytics import YOLO
data_yaml = "../dataset/data.yaml"
model_path = "../dataset/weights/best.pt"

print("Loading model from:", model_path)
print("Data YAML:", data_yaml)

model = YOLO(model_path)
results = model.val(data=data_yaml, split="val")