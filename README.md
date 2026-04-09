# Seizure Detection (Real-Time) — Project So Far

This repo is building toward a real-time seizure detection demo:

- **FastAPI** backend that ingests “EEG” samples and serves the most recent sample plus a (currently dummy) seizure probability.
- **Mock EEG streamer** that simulates **23-channel** EEG with occasional large anomalies (“mock seizures”) and POSTs samples to the API.
- **Streamlit dashboard** that polls the API and plots the live EEG and seizure probability in near real-time.

Right now, the “model” is a placeholder (random probability). The engineering scaffolding (ingest → serve → visualize) is in place.

---

## What’s in the repo

- `api.py`
  - `POST /ingest`: receives one EEG sample (23 floats)
  - `GET /latest`: returns the latest ingested sample + a random `seizure_probability`
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

## Setup

Create and activate a virtual environment, then install dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Run the system (3 terminals)

You’ll typically run **FastAPI**, the **streamer**, and the **dashboard** at the same time.

### 1) Start the API (Terminal A)

```bash
source venv/bin/activate
uvicorn api:app --reload
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
python mock_streamer.py --hz 100 --seizure-every 60 --seizure-duration 5
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

## Where this is headed (next milestones)

The long-term target (from your project spec) is:

- **Real dataset**: CHB-MIT Scalp EEG Database (PhysioNet)
- **Model**: train a CNN/LSTM (or other deep model) for seizure detection
- **Engineering twist**:
  - DVC for data/versioning
  - FastAPI serving the best model
  - Streaming simulator that mimics real-time acquisition
  - Docker containerization
  - Deployment to a free tier (Render/AWS free tier)
  - Streamlit dashboard for live visualization

When you’re ready for the next step, the most natural progression is:

- Replace the random probability in `api.py` with a real inference function.
- Add a “windowed” inference approach (e.g., model expects \(N \times 23\) samples rather than a single sample).
- Introduce DVC and a `data/` pipeline for dataset download + preprocessing.

