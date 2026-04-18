# Seizure Detection from Real-Time EEG

This repo is building toward a real-time seizure detection demo:

- **FastAPI** backend that ingests “EEG” samples and serves the most recent sample plus a (currently dummy) seizure probability.
- **Mock EEG streamer** that simulates **23-channel** EEG with occasional large anomalies (“mock seizures”) and POSTs samples to the API.
- **Streamlit dashboard** that polls the API and plots the live EEG and seizure probability in near real-time.
- Uses a custom 8 electrode EEG setup for real-time streaming
- Tested on commercially available g.Tec Unicorn (not sponsored) with custom API for real-time streaming (API not available in this repo)

This repo now supports a full **end-to-end** pipeline where you:

- train in **Colab** (PyTorch),
- export the trained model to **ONNX** (`latest.onnx`),
- upload to **Google Cloud Storage (GCS)** at a stable path,
- run the app locally (or in Docker) while the backend auto-fetches the model via a **public URL**.

---

## What’s in the repo

- `api.py`
  - `POST /ingest`: receives one EEG sample (23 floats)
  - `GET /latest`: returns the latest ingested sample + a `seizure_probability`
- `mock_streamer.py`
  - Generates a 23-channel sine-based signal
  - Every `--seizure-every` seconds, injects an anomaly for `--seizure-duration` seconds
  - Streams samples to the API via HTTP POST
- `dashboard.py`
  - Streamlit UI with Start/Stop
  - Polls `GET /latest` on an interval
  - Rolling buffer window and plots
- `requirements.txt`
  - Python dependencies (FastAPI, Uvicorn, Streamlit, etc.)
- `export_and_upload_onnx.py`
  - Optional local helper to export `GLOBAL_eeg_model_TOP10.pt` → `latest.onnx` and upload to GCS

---

## Data contract (important)

### `POST /ingest`

The API expects a JSON object with a `data` field (because it uses a Pydantic model):

```json
{
  "data": [0.12, -0.03, 0.55, "... 23 floats total ..."]
}
```

- `data` must be a list of floats (length **23** for this mock setup).

### `GET /latest`

Returns either:

```json
{"message": "No data available"}
```

or:

```json
{
  "data": [ ... 23 floats ... ],
  "seizure_probability": 0.42
}
```

---

## Run tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Tests are organized under `tests/` by pipeline stage (`stage_01_*` through `stage_07_*`).

---

## Setup

Create and activate a virtual environment, then install dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Run the system (3 terminals, local)

You’ll typically run **FastAPI**, the **streamer**, and the **dashboard** at the same time.

### 1) Start the API (Terminal A)

```bash
source venv/bin/activate
python3 -m uvicorn api:app --reload
```

FastAPI will run at `http://127.0.0.1:8000`.

- Interactive docs: `http://127.0.0.1:8000/docs`
- What you should see:
  - Terminal logs showing `Uvicorn running on http://127.0.0.1:8000`
  - If you visit `/latest` before streaming starts, you’ll get `{"message":"No data available"}`

### 2) Start the mock streamer (Terminal B)

```bash
source venv/bin/activate
python mock_streamer.py
```

You should see periodic status lines like:

- `status=200` (good)
- `seizure=True/False` depending on whether the anomaly window is active
- What you should see in the **FastAPI terminal**:
  - Repeated requests like `POST /ingest ... 200 OK`

Useful flags:

```bash
python mock_streamer.py --hz 128 --seizure-every 60 --seizure-duration 5
```

### 3) Start the Streamlit dashboard (Terminal C)

```bash
source venv/bin/activate
streamlit run dashboard.py
```

Then open the local URL Streamlit prints (commonly `http://localhost:8501`) and press **Start**.

- What you should see:
  - The Streamlit page shows **Status: Connected** once it can reach the backend
  - The EEG chart updates as samples are ingested
  - The probability chart updates (currently random, 0–1)

---

## Run with Docker (backend + frontend together)

This repo includes a `Dockerfile` and `docker-compose.yml` to run **FastAPI** + **Streamlit** together.

### 1) Build images

From the repo root:

```bash
docker compose build
```

What you should see:
- Docker building an image that installs `requirements.txt`

### 2) Start the services

```bash
docker compose up
```

Or detached:

```bash
docker compose up -d
```

What you should see:
- Backend listening on `0.0.0.0:8000`
- Frontend (Streamlit) listening on `0.0.0.0:8501`

Open:
- Streamlit: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

### 3) Start the streamer (still needed)

The compose file runs the backend + dashboard. To generate “live” data, run the streamer separately on your host:

```bash
source venv/bin/activate
python mock_streamer.py
```

What you should see:
- Streamer logs show `status=200`
- The Streamlit dashboard starts plotting live updates after you press **Start**

### 4) Stop everything

```bash
docker compose down
```

Tip: If you used `-d`, view logs with:

```bash
docker compose logs -f
```

---

## Quick verification with curl

### Confirm the API is up

```bash
curl -i http://127.0.0.1:8000/latest
```

What you should see:
- HTTP `200 OK`
- Either `{"message":"No data available"}` (before ingest) or `{"data":[...],"seizure_probability":...}`

### Manually ingest a sample (should return 200)

```bash
curl -i -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"data":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]}'
```

What you should see:
- HTTP `200 OK`
- Response: `{"message":"Data ingested successfully"}`

### Fetch the latest sample (should now include `data` and `seizure_probability`)

```bash
curl -s http://127.0.0.1:8000/latest
```

What you should see:
- A JSON object with:
  - `data`: list of floats
  - `seizure_probability`: float in `[0,1]`

---

## Common issues & fixes

### “No data available” forever

This means the API hasn’t successfully received any samples.

- Make sure the streamer is running and returning `status=200`.
- Check the FastAPI terminal for `POST /ingest ... 200 OK`.

### `POST /ingest` returns **422 Unprocessable Entity**

This is almost always a **payload shape mismatch**.

The API expects:

```json
{"data": [ ... ]}
```

If you accidentally send a raw list like `[ ... ]`, FastAPI will reject it with 422.

### Streamlit “opens too many figures”

If you see warnings about too many Matplotlib figures, the dashboard should be closing figures after rendering. This repo’s `dashboard.py` does that (`plt.close(fig)`).

---

## Model deployment (step-by-step)

This is the workflow that makes the portfolio demo “always run”.

### Step 0: Know what the backend expects

- **Model format**: ONNX (`latest.onnx`)
- **Model input shape**: `(1, 10, 256)` where:
  - `10` = “best channels” from your notebook: `[0, 1, 5, 2, 13, 18, 14, 8, 19, 16]`
  - `256` = 2 seconds @ 128Hz
- **Streaming sample rate**: set streamer to `--hz 128` so the backend accumulates one 2s window in ~2 seconds.

### Step 1: Export ONNX in Colab

Open `Seizure_Detection.ipynb` and run the cells that:

- train and save your weights to Drive (your notebook saves `GLOBAL_eeg_model_TOP10.pt`)
- **export to ONNX** (the notebook now writes):
  - `/content/drive/MyDrive/EEG_Seizure_Project/latest.onnx`

### Step 2: Upload ONNX to GCS (stable path)

In the notebook, run the auth cell (`auth.authenticate_user()`), then run the upload cell that writes to:

- `gs://YOUR_BUCKET/models/latest.onnx`

This stable path is important: every retrain/upload overwrites the same object, so your app can keep using the same URL.

### Step 3: Make the model public (read-only)

You have two ways:

- **Bucket-level public read** (simple for portfolios):

```bash
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET \
  --member=allUsers \
  --role=roles/storage.objectViewer
```

- **Object-level ACL** (may be blocked if Public Access Prevention is enabled on the bucket).

### Step 4: Get the public URL

For a bucket `YOUR_BUCKET` and object `models/latest.onnx`, the public URL is:

- `https://storage.googleapis.com/YOUR_BUCKET/models/latest.onnx`

### Step 5: Point the backend at the model URL

Run the backend with:

```bash
export MODEL_URL="https://storage.googleapis.com/YOUR_BUCKET/models/latest.onnx"
export MODEL_REFRESH_SECONDS="0"   # only download at startup (simple)
python3 -m uvicorn api:app --reload
```

Optional: set `MODEL_REFRESH_SECONDS="300"` (5 min) if you want periodic refresh.

### Step 6: Run streamer + dashboard

Streamer:

```bash
python3 mock_streamer.py --hz 128
```

Dashboard:

```bash
streamlit run dashboard.py
```

### Step 7: Verify the model is actually in GCS (avoids NoSuchKey)

If you get:

```xml
<Code>NoSuchKey</Code>
```

it means the object path doesn’t exist (upload didn’t happen or went to a different path).

Check:

```bash
gcloud storage ls gs://YOUR_BUCKET/models/
```

You must see `latest.onnx` there.

### Step 8: Verify it’s public (avoids 403)

Open the URL in an incognito window or run:

```bash
curl -I "https://storage.googleapis.com/YOUR_BUCKET/models/latest.onnx"
```

- `200` = good
- `403` = object exists but is not public (bucket policy / public access prevention)

---
---
## Where this is headed (next milestones)

- **Retraining Model**: On other datasets (PhysioNet/Kaggle)
- **Explore other Model**: train for seizure detection
- **Shadow Deployment**: Making it as free/cost-efficient as possible
---
---

## Model weights: auto-fetch `latest.onnx` (portfolio-friendly)

This repo now supports an **end-to-end** workflow where you keep training in Colab and your running app can **auto-fetch** the newest model artifact.

For deployment portability (and to avoid huge `torch` installs), the backend is set up to run **ONNX** with `onnxruntime` (CPU).

### Recommended approach (simple + reliable): Public GCS URL

1) In Colab, export your trained PyTorch model to ONNX (stable input shape: `1 x 10 x 256`), producing:

- `/content/drive/MyDrive/EEG_Seizure_Project/latest.onnx`

2) Upload your model to a stable object path:

```bash
gcloud storage cp "/content/drive/MyDrive/EEG_Seizure_Project/latest.onnx" "gs://YOUR_BUCKET/models/latest.onnx"
```

3) Make the object publicly readable (read-only).

4) Run the backend with:

- `MODEL_URL`: public URL, e.g. `https://storage.googleapis.com/YOUR_BUCKET/models/latest.onnx`
- `MODEL_LOCAL_PATH`: where to cache locally (default: `artifacts/model.onnx`)
- `MODEL_REFRESH_SECONDS`: optional periodic refresh interval (default: `0` = only at startup)

Example:

```bash
export MODEL_URL="https://storage.googleapis.com/YOUR_BUCKET/models/latest.onnx"
export MODEL_REFRESH_SECONDS="300"
uvicorn api:app --reload
```

Notes:
- The backend does **real inference** once it has accumulated a full rolling window (\(2s\) at \(128Hz\) = \(256\) samples) and the ONNX model is loaded.
- Until the buffer is full (or if the model isn’t configured), `seizure_probability` returns `0.0` so the dashboard keeps working.
- The local cache folder is tracked via `artifacts/.gitkeep` but the weights themselves should not be committed.

### Local (non-Colab) export + upload helper

If you want a local script version of the “export to ONNX + upload to GCS” step, use `export_and_upload_onnx.py`.

Example:

```bash
python3 export_and_upload_onnx.py \
  --pt GLOBAL_eeg_model_TOP10.pt \
  --onnx latest.onnx \
  --bucket YOUR_BUCKET \
  --object models/latest.onnx \
  --make-public
```
