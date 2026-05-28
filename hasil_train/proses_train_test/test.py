from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================

model = YOLO("best.pt")

# =========================
# PREDICT
# =========================

results = model.predict(
    source="31.jpg",
    save=True,
    conf=0.65
)

# =========================
# TAMPILKAN HASIL DETEKSI
# =========================

print("\nHASIL DETEKSI:\n")

for r in results:

    for box in r.boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]

        print(f"Warna : {class_name}")
        print(f"Confidence : {confidence:.2f}")
        print(f"Posisi : ({int(x1)}, {int(y1)})")
        print("----------------------")

print("\nDeteksi selesai")