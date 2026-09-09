from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="FARMORA Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("api/best.pt") 

INFERENCE_IMGSZ = 960  

CLASS_CONFIG = {
    "cyperus": {
        "display_name": "Rumput Teki",
        "min_conf": 0.40,
    },
    "Amaranthus-spinosus": {
        "display_name": "Bayam Duri",
        "min_conf": 0.28,
    },
}


def severity_from_confidence(conf: float) -> str:
    if conf >= 0.6:
        return "Tinggi"
    elif conf >= 0.4:
        return "Sedang"
    return "Rendah"



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash"

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


async def generate_summary(detections: list) -> str:
    """
    Generate an action-recommendation paragraph (Bahasa Indonesia) via Gemini Flash,
    without restating raw model output (species/confidence/severity).
    Falls back to a static message if no API key is configured or the call fails.
    """
    if not gemini_client:
        return "Rekomendasi AI tidak tersedia (GEMINI_API_KEY belum dikonfigurasi)."

    if not detections:
        return "Tidak ada gulma yang terdeteksi, tidak ada tindakan pengendalian yang perlu dilakukan saat ini."

    weed_names = ", ".join(sorted({d["display_name"] for d in detections}))

    prompt = (
        "Kamu adalah asisten pertanian yang berperan sebagai explainer tindakan, "
        f"bukan penyampai hasil deteksi. Gulma berikut terdeteksi pada lahan: {weed_names}.\n\n"
        "Berdasarkan itu, tuliskan rekomendasi tindakan lanjutan dalam Bahasa Indonesia "
        "(maksimal 4 kalimat) yang menjelaskan apa yang perlu dilakukan petani selanjutnya "
        "(misalnya cara pengendalian, waktu penanganan, atau langkah pencegahan). "
        "JANGAN menyebutkan nama gulma teknis, tingkat kepercayaan (confidence), tingkat "
        "keparahan (severity), atau hasil mentah lain dari model deteksi — fokus hanya pada "
        "tindakan yang harus dilakukan."
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception:
        return "Gulma terdeteksi pada lahan. Segera lakukan pengendalian gulma secara manual atau kimiawi sesuai kondisi lahan."


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model.predict(
        img,
        conf=0.20,       
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

        detections.append({
            "class": class_name,
            "display_name": config["display_name"],
            "confidence": round(confidence, 3),
            "bbox": xywh,
            "severity": severity_from_confidence(confidence),
        })

    summary = await generate_summary(detections)

    return {"detections": detections, "count": len(detections), "summary": summary}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}