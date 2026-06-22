# wildfire_ai_alert

MVP wildfire early-warning prototype using a modular multi-model vision pipeline:

1. **GroundingDINO**: text-guided candidate detection (`smoke`, `flame`, `fire`, `smoke plume`)
2. **SAM2**: candidate-region segmentation from detection boxes
3. **VLM explainer (placeholder)**: scene interpretation + false-positive reasoning
4. **Rule engine**: LOW / MEDIUM / HIGH risk scoring and alert generation

The code is designed to run end-to-end even without model weights via `--mock`.

---

## Architecture

Video input -> frame sampler -> preprocessing -> GroundingDINO -> SAM2 -> temporal analysis -> VLM explanation -> risk engine -> overlay + alerts.json

- GroundingDINO finds **where smoke/flame candidates might be** from text prompts.
- SAM2 refines those coarse boxes into **pixel-level masks**.
- A VLM (or placeholder) interprets context to reduce false positives (cloud/fog/dust) and generate human-readable evidence.
- Rule engine combines confidence + mask area + temporal persistence + mask growth to produce actionable alert levels.

---

## Project Structure

```text
wildfire_ai_alert/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── main.py
│   ├── input/
│   │   └── video_loader.py
│   ├── models/
│   │   ├── grounding_dino_wrapper.py
│   │   ├── sam2_wrapper.py
│   │   └── vlm_explainer.py
│   ├── pipeline/
│   │   ├── wildfire_pipeline.py
│   │   ├── preprocessing.py
│   │   ├── postprocessing.py
│   │   └── risk_engine.py
│   └── utils/
│       ├── visualization.py
│       └── logging_utils.py
└── outputs/
    ├── frames/
    ├── overlays/
    └── alerts.json
```

---

## Setup

```bash
cd wildfire_ai_alert
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run

### 1) Mock mode (recommended first)

Runs full pipeline without GroundingDINO/SAM2 checkpoints.

```bash
python -m src.main --input path/to/video.mp4 --output outputs/ --mock
```

### 1-b) Profiling + bottleneck analysis

Enable stage-level timing instrumentation and bottleneck reporting:

```bash
python -m src.main --input path/to/video.mp4 --output outputs/ --mock --profile
```

or with real backends:

```bash
python -m src.main --input path/to/video.mp4 --output outputs/ --profile
```

When `--profile` is enabled, the pipeline records stage latencies (ms) for:

- `frame_load`
- `preprocessing`
- `grounding_dino_inference`
- `sam2_inference`
- `postprocessing`
- `temporal_tracking`
- `vlm_explanation`
- `risk_engine`
- `visualization`
- `json_write`
- `total_per_frame`

And writes:

- `outputs/performance.json`: per-frame timings
- `outputs/performance_summary.json`: aggregate stats (`avg`, `min`, `max`, `p50`, `p90`, `p95`, `p99`) + auto bottleneck stage

Console summary is also printed:

- processed frames
- total runtime
- average FPS
- average latency per frame
- bottleneck stage

> Note on mock mode timings:
> In `--mock` mode, profiling is mainly useful for validating pipeline overhead, orchestration, and I/O cost.
> For true model latency benchmarking, run with real GroundingDINO/SAM2/VLM inference backends and actual checkpoints.

### 2) Real-model mode (GroundingDINO + SAM2)

```bash
python -m src.main --input path/to/video.mp4 --output outputs/
```

If required files are missing, the pipeline exits with a clear error that includes:

- which model file path is missing, and
- guidance to run with `--mock`.

---

## Real Model Setup

Install optional real-inference dependencies first:

```bash
pip install torch torchvision
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/sam2.git
```

### 1) Download GroundingDINO weights

1. Download a GroundingDINO checkpoint (example: `groundingdino_swint_ogc.pth`).
2. Locate its matching model config file from the GroundingDINO repo (example: `GroundingDINO_SwinT_OGC.py`).

### 2) Download SAM2 weights

1. Download a SAM2 checkpoint (example: `sam2_hiera_large.pt`).
2. Locate the matching SAM2 model config YAML (example: `sam2_hiera_l.yaml`).

### 3) Configure `config.yaml` paths

```yaml
models:
  device: null  # or "cpu" / "cuda"
  grounding_dino:
    config: "/absolute/path/to/GroundingDINO_SwinT_OGC.py"
    checkpoint: "/absolute/path/to/groundingdino_swint_ogc.pth"
    box_threshold: 0.35
    text_threshold: 0.25
  sam2:
    model_config: "/absolute/path/to/sam2_hiera_l.yaml"
    checkpoint: "/absolute/path/to/sam2_hiera_large.pt"
```

### 4) Run with real backends

```bash
python -m src.main --input path/to/video.mp4 --output outputs/ --profile
```

If you only want to validate pipeline flow without model files, run:

```bash
python -m src.main --input path/to/video.mp4 --output outputs/ --mock --profile
```

---

## Outputs

- `outputs/frames/`: sampled raw frames
- `outputs/overlays/`: detection boxes + mask + risk label overlays
- `outputs/alerts.json`: per-frame risk records
- `outputs/performance.json`: per-frame profiling records (when `--profile`)
- `outputs/performance_summary.json`: profiling summary + bottleneck (when `--profile`)

`alerts.json` fields:
- `frame_id`
- `timestamp`
- `detected_labels`
- `confidence`
- `mask_area_ratio`
- `mask_growth`
- `risk_level`
- `explanation`
- `alert_message`

---

## Why GroundingDINO + SAM2?

GroundingDINO provides flexible **text-conditioned localization** of wildfire cues (`smoke`, `flame`, etc.), while SAM2 converts those coarse boxes into **precise segmentation** for area-based and temporal risk analytics. This pairing improves interpretability and robustness versus box-only detections.

## Why VLM after segmentation?

Segmentation alone cannot fully resolve ambiguity (e.g., cloud/fog/dust/sun glare). A VLM layer can reason over scene context and temporal evidence, helping reduce false positives and produce operator-friendly explanations.

---

## Future Notes: RBLN NPU Deployment

- Add RBLN-compiled model paths/config under `config.yaml`.
- Replace wrappers with RBLN runtime backends while preserving current interfaces.
- Keep rule engine and output schema unchanged to maintain operational continuity.
- Benchmark NPU latency/throughput and compare against GPU baseline in the same pipeline.

