# FARMORA — Weed Detection API

A YOLOv8-based object detection service for identifying weeds in crop fields, built for the FARMORA project. It detects two weed species — *Cyperus rotundus* (Rumput Teki) and *Amaranthus spinosus* (Bayam Duri) — and returns bounding boxes, confidence-based severity, and herbicide dosage recommendations via a FastAPI endpoint.

## Project structure

```
api/            FastAPI inference server (main.py) + copied model weights (best.pt)
scripts/        Data preparation, training, and local testing scripts
datasets/       Merged, remapped dataset used for training (generated, gitignored)
raw_bayam_duri/     Raw Roboflow dataset for Amaranthus spinosus (gitignored)
raw_rumput_teki/    Raw Roboflow dataset for Cyperus rotundus (gitignored)
runs/           Ultralytics training/prediction outputs (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics fastapi uvicorn python-dotenv pillow roboflow
```

Create a `.env` file in the project root:

```
ROBOFLOW_API_KEY=your_roboflow_api_key
```

## Preparing data

Downloads the raw datasets from Roboflow, remaps class IDs, and merges them into `datasets/merged`:

```bash
python scripts/prepare_data.py
```

## Training

```bash
python scripts/train.py
```

Trains a YOLOv8s model at 960px using `scripts/data.yaml`. Outputs are written to `runs/detect/`.

## Local prediction test

```bash
python scripts/test.py
```

Runs inference on a sample image with `scripts/test.py`, adjust the image path and thresholds as needed.

## Running the API

The API expects trained weights at `api/best.pt` (or update the path in `api/main.py`), then:

```bash
cd api
uvicorn main:app --reload
```

### Endpoints

- `POST /detect` — upload an image (`multipart/form-data`, field `image`), returns detected weeds with class, confidence, bounding box, severity, and recommended treatment.
- `GET /health` — health check.

## Detected classes

| Class ID | Class name             | Display name  | Min. confidence |
|----------|-------------------------|---------------|------------------|
| 0        | `cyperus`               | Rumput Teki   | 0.40             |
| 1        | `Amaranthus-spinosus`   | Bayam Duri    | 0.28             |
