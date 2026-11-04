from ultralytics import YOLO
import os

if __name__ == "__main__":

    # ---------------------------
    # Load pre trained model
    # ---------------------------
    model = YOLO("pretrained.pt")

    # ---------------------------
    # Dataset YAML file
    # ---------------------------
    data_yaml = os.path.abspath(r"..\dataset\data.yaml")

    # ---------------------------
    # Train model
    # ---------------------------
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        patience=20,
        optimizer="auto",
        project=r"..\detection\training_results",
        name="YOLO11s_DeepSea_Net",
        pretrained=True,
        save=True,
        plots=True,
        verbose=True
    )




