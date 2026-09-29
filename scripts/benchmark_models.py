"""Phase 4 benchmark: for each registered encoder, measure parameter
count, checkpoint size, approximate compute, CPU/GPU model-only latency
(P50/P95, warm-up first), peak GPU memory, image batch inference, and
fixed 4/8/16-frame video inference. Prints a provisional efficiency
comparison - NOT a final model selection, since no real deepfake accuracy
result exists yet (see docs/PROJECT_PLAN.md Phase 4 scope).

Usage:
    .venv/Scripts/python.exe scripts/benchmark_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")  # MUST run before importing timm/torch/huggingface_hub

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configuard.models.benchmark import approximate_flops, measure_gpu_peak_memory, measure_latency  # noqa: E402
from configuard.models.device import resolve_device  # noqa: E402
from configuard.models.inference import infer_image_batch, infer_video_fixed_frames  # noqa: E402
from configuard.models.registry import create_encoder, list_encoder_names  # noqa: E402


def _random_frames(n: int, size: int = 224) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]


def main() -> int:
    cpu = torch.device("cpu")
    gpu = resolve_device("cuda") if torch.cuda.is_available() else None

    print("=== ConfiGuard-Lite Phase 4 baseline model benchmark ===")
    print(f"CPU: available. GPU: {'available (' + torch.cuda.get_device_name(0) + ')' if gpu else 'not available'}\n")

    results = {}
    for name in list_encoder_names():
        print(f"########## {name} ##########")
        encoder = create_encoder(name, pretrained=True)
        param_count = encoder.parameter_count()
        trainable = encoder.trainable_parameter_count()
        preprocess_cfg = encoder.resolve_preprocess_config()

        print(f"parameters            : {param_count:,}")
        print(f"trainable parameters  : {trainable:,}")

        flops = approximate_flops(encoder, input_size=224, device="cpu")
        print(f"approx FLOPs @224x224 : {flops:,}" if flops is not None else "approx FLOPs @224x224 : not available")

        # --- CPU latency (single image, batch=1) ---
        encoder_cpu = encoder.to(cpu).eval()
        x_cpu = torch.randn(1, 3, 224, 224)
        cpu_report = measure_latency(lambda: encoder_cpu(x_cpu), warmup=5, iters=20, device=cpu)
        print(f"CPU latency (bs=1)    : P50={cpu_report.p50_ms:.2f}ms P95={cpu_report.p95_ms:.2f}ms "
              f"(warmup={cpu_report.warmup_iters}, n={cpu_report.measured_iters})")

        # --- Image batch inference (CPU) ---
        frames_batch = _random_frames(8)
        batch_report = measure_latency(
            lambda: infer_image_batch(encoder_cpu, frames_batch, preprocess_cfg, device=cpu),
            warmup=3, iters=10, device=cpu,
        )
        print(f"CPU image batch (n=8) : P50={batch_report.p50_ms:.2f}ms P95={batch_report.p95_ms:.2f}ms")

        # --- Fixed 4/8/16-frame video inference (CPU) ---
        video_latencies = {}
        for frame_count in (4, 8, 16):
            frames = _random_frames(frame_count)
            report = measure_latency(
                lambda f=frames: infer_video_fixed_frames(encoder_cpu, f, preprocess_cfg, device=cpu),
                warmup=3, iters=10, device=cpu,
            )
            video_latencies[frame_count] = report
            print(f"CPU video-{frame_count:>2} latency  : P50={report.p50_ms:.2f}ms P95={report.p95_ms:.2f}ms")

        gpu_report = None
        gpu_mem = None
        if gpu is not None:
            encoder_gpu = encoder.to(gpu).eval()
            x_gpu = torch.randn(1, 3, 224, 224, device=gpu)
            gpu_report = measure_latency(lambda: encoder_gpu(x_gpu), warmup=10, iters=30, device=gpu)
            print(f"GPU latency (bs=1)    : P50={gpu_report.p50_ms:.2f}ms P95={gpu_report.p95_ms:.2f}ms "
                  f"(warmup={gpu_report.warmup_iters}, n={gpu_report.measured_iters})")

            gpu_mem = measure_gpu_peak_memory(lambda: encoder_gpu(x_gpu), gpu)
            print(f"GPU peak memory       : allocated={gpu_mem.peak_allocated_mb:.1f}MB "
                  f"reserved={gpu_mem.peak_reserved_mb:.1f}MB "
                  f"({gpu_mem.peak_reserved_mb / 1024:.3f} GB of 6 GB budget)")
            encoder.to(cpu)  # free GPU memory before the next model

        results[name] = {
            "param_count": param_count, "flops": flops,
            "cpu_p50_ms": cpu_report.p50_ms, "cpu_p95_ms": cpu_report.p95_ms,
            "gpu_p50_ms": gpu_report.p50_ms if gpu_report else None,
            "gpu_p95_ms": gpu_report.p95_ms if gpu_report else None,
            "gpu_peak_mb": gpu_mem.peak_reserved_mb if gpu_mem else None,
        }
        print()

    print("=== Provisional efficiency comparison (latency/size only - NOT a final model choice) ===")
    print(f"{'model':30s} {'params':>12s} {'CPU P50 ms':>12s} {'GPU P50 ms':>12s} {'GPU peak MB':>12s}")
    for name, r in results.items():
        gpu_p50 = f"{r['gpu_p50_ms']:.2f}" if r["gpu_p50_ms"] is not None else "n/a"
        gpu_mb = f"{r['gpu_peak_mb']:.1f}" if r["gpu_peak_mb"] is not None else "n/a"
        print(f"{name:30s} {r['param_count']:>12,} {r['cpu_p50_ms']:>12.2f} {gpu_p50:>12s} {gpu_mb:>12s}")
    print(
        "\nNOTE: this is a provisional efficiency comparison only. No deepfake-detection "
        "accuracy result exists yet (both models are ImageNet-pretrained with an untrained "
        "binary head) - the final model choice must not be based on latency alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
