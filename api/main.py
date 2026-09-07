from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FARMORA Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo purposes; tighten this later if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("../runs/detect/farmora_gulma-4/weights/best.pt")  # this should be the v3 weights, copied in as api/best.pt

INFERENCE_IMGSZ = 960  # model was trained at 960, keep inference consistent

# --- Per-class confidence threshold + recommendation lookup ---
CLASS_CONFIG = {
    "cyperus": {
        "display_name": "Rumput Teki",
        "min_conf": 0.40,
        "recommendation": {"type": "Herbisida", "dose_ml": 50},
    },
    "Amaranthus-spinosus": {
        "display_name": "Bayam Duri",
        "min_conf": 0.28,
        "recommendation": {"type": "Herbisida", "dose_ml": 40},
    },
}


def severity_from_confidence(conf: float) -> str:
    if conf >= 0.6:
        return "Tinggi"
    elif conf >= 0.4:
        return "Sedang"
    return "Rendah"



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # not used yet, placeholder for later

DUMMY_EXPLANATIONS = {
    "cyperus": (
        "Terdeteksi Rumput Teki (Cyperus rotundus) dengan tingkat keyakinan {conf}%. "
        "Gulma ini dikenal sulit dikendalikan karena sistem rimpang di bawah tanah "
        "yang menyebar cepat dan bersaing kuat dengan tanaman budidaya untuk nutrisi "
        "dan air. Penanganan dini disarankan sebelum rimpang menyebar lebih luas."
    ),
    "Amaranthus-spinosus": (
        "Terdeteksi Bayam Duri (Amaranthus spinosus) dengan tingkat keyakinan {conf}%. "
        "Gulma ini memiliki duri tajam di ketiak daun sebagai ciri khasnya, tumbuh "
        "cepat, dan dapat mengganggu pertumbuhan tanaman di sekitarnya jika tidak "
        "segera ditangani."
    ),
}


async def get_gemini_explanation(class_name: str, confidence: float, image_bytes: bytes = None) -> str:
    """
    PLACEHOLDER — returns a dummy explanation string.

    Real implementation (uncomment + fill in when ready):

        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            f"Jelaskan secara singkat dalam Bahasa Indonesia mengenai gulma "
            f"{class_name} yang terdeteksi dengan confidence {confidence:.0%}, "
            f"termasuk ciri khas dan mengapa perlu dikendalikan. Maks 3 kalimat."
        )
        response = gemini_model.generate_content(prompt)
        return response.text
    """
    template = DUMMY_EXPLANATIONS.get(
        class_name,
        "Gulma terdeteksi dengan tingkat keyakinan {conf}%."
    )
    return template.format(conf=round(confidence * 100))


# =========================================================================
# DETECTION ENDPOINT
# =========================================================================

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model.predict(
        img,
        conf=0.20,       # low global floor; real filtering happens per-class below
        iou=0.4,
        imgsz=INFERENCE_IMGSZ,
        augment=True,
    )

    detections = []
    for box in results[0].boxes:
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        config = CLASS_CONFIG.get(class_name)

        if config is None:
            continue
        if confidence < config["min_conf"]:
            continue

        xywh = box.xywh.tolist()[0]  # [x_center, y_center, width, height] in pixels

        explanation = await get_gemini_explanation(class_name, confidence, contents)

        detections.append({
            "class": class_name,
            "display_name": config["display_name"],
            "confidence": round(confidence, 3),
            "bbox": xywh,
            "severity": severity_from_confidence(confidence),
            "recommendation": config["recommendation"],
            "explanation": explanation,
        })

    return {"detections": detections, "count": len(detections)}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}