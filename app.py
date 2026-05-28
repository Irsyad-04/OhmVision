from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# =========================
# CONFIG
# =========================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# LOAD MODEL
# =========================

model = YOLO("best.pt")

# =========================
# MAP DATA
# =========================

digit_map = {

    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "gray": 8,
    "grey": 8,
    "white": 9

}

multiplier_map = {

    "black": 1,
    "brown": 10,
    "red": 100,
    "orange": 1000,
    "yellow": 10000,
    "green": 100000,
    "blue": 1000000,
    "gold": 0.1,
    "silver": 0.01

}

tolerance_map = {

    "brown": "±1%",
    "red": "±2%",
    "green": "±0.5%",
    "blue": "±0.25%",
    "violet": "±0.1%",
    "gray": "±0.05%",
    "gold": "±5%",
    "silver": "±10%"

}

# =========================
# RESISTOR CALCULATION
# =========================

def calculate_resistor(colors):

    if len(colors) < 4:

        return "Unknown"

    d1 = digit_map.get(colors[0], 0)

    d2 = digit_map.get(colors[1], 0)

    multiplier = multiplier_map.get(colors[2], 1)

    value = ((d1 * 10) + d2) * multiplier

    if value >= 1000000:

        return f"{value / 1000000:.1f} MΩ"

    elif value >= 1000:

        return f"{value / 1000:.1f} kΩ"

    else:

        return f"{int(value)} Ω"

# =========================
# HOME ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])

def home():

    result_image = None
    original_image = None

    resistor_type = "-"
    result_value = "-"
    tolerance_value = "-"

    bands = []
    band_info = []
    band_confidence = []

    error_message = None

    # =====================
    # POST METHOD
    # =====================

    if request.method == "POST":

        # CHECK FILE

        if "image" not in request.files:

            return render_template(

                "index.html",

                error_message="No image uploaded"

            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(

                "index.html",

                error_message="Please select image"

            )

        try:

            # =====================
            # SAVE IMAGE
            # =====================

            upload_path = os.path.join(

                app.config["UPLOAD_FOLDER"],
                file.filename

            )

            file.save(upload_path)

            original_image = upload_path

            # =====================
            # YOLO DETECT
            # =====================

            results = model.predict(

                source=upload_path,
                conf=0.65

            )

            annotated = results[0].plot()

            # =====================
            # SAVE RESULT IMAGE
            # =====================

            result_filename = "result_" + file.filename

            result_path = os.path.join(

                app.config["UPLOAD_FOLDER"],
                result_filename

            )

            cv2.imwrite(result_path, annotated)

            result_image = result_path

            # =====================
            # DETECTION DATA
            # =====================

            detections = []

            for box in results[0].boxes:

                cls_id = int(box.cls[0])

                label = model.names[cls_id]

                conf = float(box.conf[0])

                x1 = float(box.xyxy[0][0])

                detections.append({

                    "label": label,
                    "conf": conf,
                    "x": x1

                })

            # =====================
            # SORT LEFT TO RIGHT
            # =====================

            detections = sorted(

                detections,
                key=lambda d: d["x"]

            )

            # =====================
            # REMOVE RESISTOR LABEL
            # =====================

            color_bands = []

            for d in detections:

                if d["label"] != "resistor":

                    color_bands.append(d)

            # =====================
            # GET COLOR LIST
            # =====================

            color_names = [

                d["label"]
                for d in color_bands

            ]

            bands = color_names

            # =====================
            # ORIENTATION CHECK
            # =====================

            if "gold" in bands:

                gold_index = bands.index("gold")

                if gold_index < len(bands) / 2:

                    bands.reverse()

                    color_bands.reverse()

            elif "silver" in bands:

                silver_index = bands.index("silver")

                if silver_index < len(bands) / 2:

                    bands.reverse()

                    color_bands.reverse()

            # UPDATE COLOR NAMES

            color_names = bands

            # =====================
            # RESISTOR TYPE
            # =====================

            if len(color_names) == 4:

                resistor_type = "4 Band Resistor"

            elif len(color_names) == 5:

                resistor_type = "5 Band Resistor"

            else:

                resistor_type = f"{len(color_names)} Band"

            # =====================
            # RESISTANCE VALUE
            # =====================

            result_value = calculate_resistor(

                color_names

            )

            # =====================
            # TOLERANCE
            # =====================

            if len(color_names) >= 4:

                tolerance_value = tolerance_map.get(

                    color_names[-1],
                    "-"

                )

            # =====================
            # BAND INFO
            # =====================

            for i, band in enumerate(color_names):

                # DIGIT

                if i == 0 or i == 1:

                    info = str(

                        digit_map.get(
                            band,
                            "-"
                        )

                    )

                # MULTIPLIER

                elif i == 2:

                    info = str(

                        multiplier_map.get(
                            band,
                            "-"
                        )

                    )

                # TOLERANCE

                elif i == 3:

                    info = tolerance_map.get(

                        band,
                        "-"

                    )

                else:

                    info = "-"

                band_info.append(info)

            # =====================
            # CONFIDENCE
            # =====================

            for d in color_bands:

                conf_text = f"{round(d['conf'] * 100)}%"

                band_confidence.append(conf_text)

        except Exception as e:

            error_message = str(e)

    # =====================
    # RENDER
    # =====================

    return render_template(

        "index.html",

        result_image=result_image,
        original_image=original_image,

        resistor_type=resistor_type,
        result_value=result_value,
        tolerance_value=tolerance_value,

        bands=bands,
        band_info=band_info,
        band_confidence=band_confidence,

        error_message=error_message

    )

# =========================
# DOWNLOAD RESULT
# =========================

@app.route("/download/<filename>")

def download_file(filename):

    return send_from_directory(

        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=True

    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )

