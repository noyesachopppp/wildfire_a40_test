from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "environment_name",
    "status",
    "actual_runtime",
    "detector_model",
    "detector_backend",
    "vlm_enabled",
    "vlm_event_only",
    "vlm_model",
    "vlm_max_new_tokens",
    "processed_frames",
    "frame_load_avg_ms",
    "yolo_inference_avg_ms",
    "sam2_inference_avg_ms",
    "vlm_explanation_avg_ms",
    "queue_wait_avg_ms",
    "end_to_end_p95_ms",
    "bottleneck_stage",
    "note",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_row(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    metrics = payload.get("standardized_metrics", payload)
    row = {k: metrics.get(k) for k in CSV_COLUMNS}
    row["environment_name"] = row.get("environment_name") or source_path.parent.name
    row["status"] = row.get("status") or payload.get("status") or "SUCCESS"
    row["actual_runtime"] = row.get("actual_runtime") or payload.get("actual_runtime") or "runpod_host_python"
    row["detector_model"] = row.get("detector_model") or payload.get("detector_model")
    row["detector_backend"] = row.get("detector_backend") or payload.get("detector_backend")
    row["processed_frames"] = row.get("processed_frames") or payload.get("processed_frames")
    row["vlm_enabled"] = row.get("vlm_enabled")
    row["vlm_event_only"] = row.get("vlm_event_only")
    row["vlm_model"] = row.get("vlm_model")
    row["vlm_max_new_tokens"] = row.get("vlm_max_new_tokens")
    row["frame_load_avg_ms"] = row.get("frame_load_avg_ms")
    row["yolo_inference_avg_ms"] = row.get("yolo_inference_avg_ms")
    row["sam2_inference_avg_ms"] = row.get("sam2_inference_avg_ms")
    row["vlm_explanation_avg_ms"] = row.get("vlm_explanation_avg_ms") or row.get("vlm_inference_avg_ms")
    row["queue_wait_avg_ms"] = row.get("queue_wait_avg_ms")
    row["end_to_end_p95_ms"] = row.get("end_to_end_p95_ms")
    row["bottleneck_stage"] = row.get("bottleneck_stage") or payload.get("bottleneck_stage")
    row["note"] = row.get("note") or payload.get("note") or payload.get("notes")
    return row


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _rule_based_interpretation(row: dict[str, Any]) -> str:
    if row.get("status") == "NOT_MEASURED" and row.get("environment_name") == "docker_yolov8n_vlm":
        return (
            "Docker scenario was planned to validate container reproducibility and container I/O overhead, "
            "but true container runtime performance was not measured because Docker runtime was unavailable."
        )

    frame_load = _as_float(row.get("frame_load_avg_ms"))
    yolo = _as_float(row.get("yolo_inference_avg_ms"))
    vlm = _as_float(row.get("vlm_explanation_avg_ms"))
    queue_wait = _as_float(row.get("queue_wait_avg_ms"))
    sam2 = _as_float(row.get("sam2_inference_avg_ms"))
    candidates = {
        "frame_load_avg_ms": frame_load,
        "yolo_inference_avg_ms": yolo,
        "vlm_explanation_avg_ms": vlm,
        "queue_wait_avg_ms": queue_wait,
        "sam2_inference_avg_ms": sam2,
    }
    valid = {k: v for k, v in candidates.items() if v is not None}
    if not valid:
        return "No dominant bottleneck rule triggered from current metrics."
    largest = max(valid, key=valid.get)

    if largest == "frame_load_avg_ms":
        return (
            "Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, "
            "separate decode pipeline, frame buffer, and storage I/O optimization."
        )
    if largest == "yolo_inference_avg_ms":
        return (
            "Primary bottleneck: detector inference. Response: use YOLOv8n as edge default, tune input "
            "resolution, frame stride, and batching."
        )
    if largest == "vlm_explanation_avg_ms":
        return (
            "Primary bottleneck: VLM explanation. Response: MEDIUM/HIGH-only invocation, max_new_tokens limit, "
            "timeout, fallback template, and async execution."
        )
    if largest == "queue_wait_avg_ms" and (queue_wait or 0.0) > 0:
        return (
            "Operational bottleneck: queue wait / worker saturation. Response: scale SAM2/VLM workers, "
            "priority queue, backpressure, timeout, and fallback."
        )
    return "No dominant bottleneck rule triggered from current metrics."


def _executed_label(rows: list[dict[str, Any]], target: str) -> str:
    for row in rows:
        if row.get("status") != "SUCCESS":
            continue
        env = str(row.get("environment_name", "")).lower()
        if target == "yolo" and "yolo" in env:
            return "Yes"
        if target == "sam2" and _as_float(row.get("sam2_inference_avg_ms")) not in (None, 0.0):
            return "Yes"
        if target == "vlm" and _as_float(row.get("vlm_explanation_avg_ms")) not in (None, 0.0):
            return "Yes"
    return "No"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "experiments" / "results"
    summary_paths = sorted(results_root.glob("*/performance_summary.json"))
    rows = [_extract_row(_load_json(path), path) for path in summary_paths]

    csv_path = results_root / "bottleneck_comparison.csv"
    md_path = results_root / "bottleneck_comparison.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines: list[str] = [
        "# Bottleneck Comparison Report",
        "",
        "## Executive Summary",
        "- This validation uses RBLN Model Zoo / RBLN-documented model candidates at the architecture level.",
        "- Current measurements are RunPod host / GPU / mock measurements, not final RBLN NPU runtime measurements.",
        "- Docker is counted only when `actual_runtime=local_docker_container`.",
        "- K8s-style split-service simulation is used for queue/worker operational bottleneck analysis.",
        "- Final deployment still requires RBLN Profiler validation.",
        "",
        "## Model Candidates",
        "| Model Candidate | Role | Executed In Current Experiments |",
        "|---|---|---|",
        f"| YOLOv8n / YOLOv8m | Detection | {_executed_label(rows, 'yolo')} |",
        f"| SAM2 | Segmentation | {_executed_label(rows, 'sam2')} |",
        f"| Qwen2.5-VL-7B | VLM Explanation | {_executed_label(rows, 'vlm')} |",
        "| A.X-4.0-Light | Optional LLM Summary | No (architecture candidate only) |",
        "",
        "## Experiment Comparison",
        "| Experiment | Status | Actual Runtime | Detector | VLM Included | Frame Load Avg | YOLO Avg | SAM2 Avg | VLM Avg | Queue Wait Avg | E2E p95 | Bottleneck Stage | Note |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for row in rows:
        vlm_included = "Yes" if row.get("vlm_enabled") else "No"
        lines.append(
            f"| {row.get('environment_name')} | {row.get('status')} | {row.get('actual_runtime')} | "
            f"{row.get('detector_model')} ({row.get('detector_backend')}) | {vlm_included} | "
            f"{row.get('frame_load_avg_ms')} | {row.get('yolo_inference_avg_ms')} | "
            f"{row.get('sam2_inference_avg_ms')} | {row.get('vlm_explanation_avg_ms')} | "
            f"{row.get('queue_wait_avg_ms')} | {row.get('end_to_end_p95_ms')} | "
            f"{row.get('bottleneck_stage')} | {row.get('note')} |"
        )

    runpod_rows = [r for r in rows if str(r.get("environment_name", "")).startswith("runpod_")]
    local_docker_rows = [r for r in rows if str(r.get("environment_name", "")).startswith("local_docker_")]
    split_rows = [r for r in rows if "split_service" in str(r.get("environment_name", ""))]

    lines.extend(
        [
            "",
            "## Validation Scope Summary",
            "### 1) RunPod GPU Host Baseline",
            "- Purpose: model + pipeline runtime bottleneck in RunPod host environment.",
            "- Metrics focus: frame_load, yolo_inference, sam2_inference, vlm_explanation.",
        ]
    )
    for row in runpod_rows:
        lines.append(
            f"- `{row.get('environment_name')}`: bottleneck=`{row.get('bottleneck_stage')}`, "
            f"frame_load_avg_ms={row.get('frame_load_avg_ms')}, yolo_avg_ms={row.get('yolo_inference_avg_ms')}, "
            f"vlm_avg_ms={row.get('vlm_explanation_avg_ms')}"
        )

    lines.extend(
        [
            "",
            "### 2) Local Docker Container Validation",
            "- Purpose: container reproducibility and container I/O overhead locally.",
            "- Interpretation rule: do not compare absolute latency with RunPod because hardware/storage differ.",
        ]
    )
    for row in local_docker_rows:
        lines.append(
            f"- `{row.get('environment_name')}`: status=`{row.get('status')}`, actual_runtime=`{row.get('actual_runtime')}`, "
            f"container bottleneck=`{row.get('bottleneck_stage')}`"
        )

    lines.extend(
        [
            "",
            "### 3) Local K8s-style Operational Simulation",
            "- Purpose: queue/worker/backpressure bottleneck under split-service separation.",
            "- Metrics focus: queue_wait, queue_depth, worker utilization, timeout/fallback.",
        ]
    )
    for row in split_rows:
        lines.append(
            f"- `{row.get('environment_name')}`: queue_wait_avg_ms={row.get('queue_wait_avg_ms')}, "
            f"bottleneck=`{row.get('bottleneck_stage')}`"
        )

    lines.extend(
        [
            "",
            "## Bottleneck Interpretation",
        ]
    )
    for row in rows:
        lines.append(f"- **{row.get('environment_name')}**: {_rule_based_interpretation(row)}")

    lines.extend(
        [
            "",
            "## Mitigation Strategy",
            "- I/O bottleneck (`frame_load`) dominates: prioritize async prefetch, decode pipeline split, and storage tuning.",
            "- Detector bottleneck dominates: keep YOLOv8n as edge default and tune resolution/stride before upscaling model.",
            "- VLM bottleneck dominates: keep MEDIUM/HIGH event-gating, cap max tokens, enforce timeout/fallback.",
            "- Queue bottleneck dominates: scale SAM2/VLM workers and apply priority/backpressure control.",
            "",
            "## Profiling Notes",
            "- Forced MEDIUM/HIGH event mode is for load/bottleneck profiling only and not for accuracy evaluation.",
            "- RunPod and local Docker latency values should not be interpreted as absolute cross-environment comparisons.",
            "",
            "## FDE Deployment Interpretation",
            "- RunPod host experiments identify model/I/O bottlenecks under current host/GPU assumptions.",
            "- Docker container experiments identify container reproducibility and container I/O overhead only when actual runtime is true container execution.",
            "- Split-service/K8s-style simulation identifies queue wait, worker saturation, and backpressure risks.",
            "- Final RBLN NPU deployment requires RBLN Profiler to verify p50/p95/p99 latency and bottleneck movement.",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()

