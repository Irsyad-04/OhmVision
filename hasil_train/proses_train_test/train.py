from ultralytics import YOLO

if __name__ == "__main__":

    # Load model
    model = YOLO("yolo11s.pt")

    # Training
    model.train(
        data="data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        patience=30,
        cache=True
    )