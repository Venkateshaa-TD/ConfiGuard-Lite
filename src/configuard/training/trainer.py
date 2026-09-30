"""The training loop: AMP, gradient accumulation/clipping, early
stopping, best/latest checkpoints, exact resume, structured logging,
non-finite-loss and CUDA-OOM handling.

Exact resume: checkpoints are only written at epoch boundaries, and
every epoch's data order is derived from (seed + epoch) rather than
from a generator that advanced through earlier epochs - so a run resumed
from "latest" sees exactly the same batches, RNG streams, optimizer/
scheduler/scaler state, and early-stopping state as an uninterrupted run
(verified bit-exact on CPU in tests/training/test_trainer.py).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.inference import compute_logits_for_batch
from configuard.training.checkpoint import (
    Checkpoint,
    CheckpointProvenance,
    build_checkpoint,
    get_git_commit,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
    verify_checkpoint_compatible,
)
from configuard.training.config import TrainingConfig
from configuard.training.logging_utils import ExperimentLogger, LogRecord
from configuard.training.metrics import ValidationMetrics, compute_validation_metrics
from configuard.training.optim import build_optimizer, build_warmup_cosine_scheduler


class NonFiniteLossError(Exception):
    """Raised when the training loss becomes NaN/Inf - training stops
    immediately, before any backward pass, optimizer step or checkpoint
    save, rather than continuing with corrupted gradients (task 17)."""


class TrainingOutOfMemoryError(Exception):
    """Raised on a CUDA OOM during a training step. The batch size is
    NEVER silently changed (task 18) - the caller must adjust the config
    and restart."""


@dataclass
class TrainState:
    epoch: int = 0  # next epoch to run
    global_step: int = 0  # optimizer step attempts (incl. AMP-skipped ones; drives accumulation/log cadence)
    skipped_optimizer_steps: int = 0  # attempts GradScaler skipped for inf/nan grads - no parameter update
    best_metric: float | None = None  # higher is better (see _selection_value)
    best_metric_name: str = ""
    best_tiebreak: float | None = None  # -val_loss of the best epoch; breaks exact ties (e.g. saturated AUROC == 1.0)
    best_epoch: int | None = None
    epochs_without_improvement: int = 0


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float | None
    val_metrics: ValidationMetrics | None
    epoch_time_seconds: float
    train_time_seconds: float
    val_time_seconds: float
    mean_step_time_seconds: float  # per micro-batch: forward + backward (+ optimizer step when due)
    peak_gpu_memory_mb: float | None  # torch.cuda.max_memory_allocated over the epoch
    peak_gpu_reserved_mb: float | None  # torch.cuda.max_memory_reserved - what nvidia-smi roughly sees
    selection_metric_name: str
    selection_value: float
    is_best: bool


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        encoder: DeepfakeVisualEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        checkpoint_dir: str | Path,
        output_dir: str | Path,
        manifest_checksum: str,
        device: torch.device,
        preprocessing_version: str = "unknown",
        val_manifest_checksum: str | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.manifest_checksum = manifest_checksum
        self.val_manifest_checksum = val_manifest_checksum
        self.preprocessing_version = preprocessing_version

        self.optimizer = build_optimizer(self.encoder, config.lr, config.weight_decay, config.freeze_backbone)
        steps_per_epoch = max(1, len(train_loader))
        optimizer_steps_per_epoch = -(-steps_per_epoch // max(1, config.grad_accum_steps))  # ceil
        total_optimizer_steps = max(1, optimizer_steps_per_epoch * max(1, config.epochs))
        self.scheduler = build_warmup_cosine_scheduler(self.optimizer, config.warmup_steps, total_optimizer_steps)

        self.use_amp = config.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, init_scale=config.amp_init_scale, enabled=self.use_amp)

        self.state = TrainState()
        self.logger = ExperimentLogger(self.output_dir, config.run_name)

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------
    def _provenance(self) -> CheckpointProvenance:
        return CheckpointProvenance(
            encoder_name=self.config.encoder_name,
            model_id=self.encoder.spec.hf_repo_id,
            model_revision=self.encoder.spec.revision,
            preprocessing_version=self.preprocessing_version,
            manifest_checksum=self.manifest_checksum,
            git_commit=get_git_commit(),
            config=self.config.to_dict(),
            val_manifest_checksum=self.val_manifest_checksum,
        )

    def checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{self.config.run_name}_{name}.pt"

    def save(self, name: str, metrics: dict[str, Any]) -> Path:
        checkpoint = build_checkpoint(
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            model=self.encoder,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.use_amp else None,
            provenance=self._provenance(),
            metrics=metrics,
            train_state=asdict(self.state),
        )
        return save_checkpoint_atomic(checkpoint, self.checkpoint_path(name))

    def resume(self, path: str | Path) -> Checkpoint:
        # Load on CPU: model/optimizer load_state_dict copy onto the
        # parameters' device, and RNG states must stay CPU tensors.
        checkpoint = load_checkpoint(path, map_location="cpu")
        verify_checkpoint_compatible(
            checkpoint,
            encoder_name=self.config.encoder_name,
            model_id=self.encoder.spec.hf_repo_id,
            preprocessing_version=self.preprocessing_version,
            manifest_checksum=self.manifest_checksum,
            val_manifest_checksum=self.val_manifest_checksum,
            config=self.config.to_dict(),
        )
        self.encoder.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        if checkpoint.scheduler_state is not None:
            self.scheduler.load_state_dict(checkpoint.scheduler_state)
        if checkpoint.scaler_state is not None and self.use_amp:
            self.scaler.load_state_dict(checkpoint.scaler_state)
        restore_rng_state(checkpoint.rng_state)
        self.state = TrainState(**checkpoint.train_state) if checkpoint.train_state else TrainState(
            epoch=checkpoint.epoch, global_step=checkpoint.global_step
        )
        return checkpoint

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def _oom_error(self, exc: BaseException, batch_size: int, phase: str) -> TrainingOutOfMemoryError:
        summary = "n/a"
        if self.device.type == "cuda":
            summary = (
                f"allocated={torch.cuda.memory_allocated(self.device) / 2**20:.0f} MB, "
                f"peak_allocated={torch.cuda.max_memory_allocated(self.device) / 2**20:.0f} MB, "
                f"reserved={torch.cuda.memory_reserved(self.device) / 2**20:.0f} MB"
            )
        return TrainingOutOfMemoryError(
            f"CUDA out of memory during {phase} (batch_size={batch_size}, "
            f"grad_accum_steps={self.config.grad_accum_steps}, amp={self.use_amp}, device={self.device}; {summary}). "
            "Batch size was NOT changed automatically - reduce batch_size (and raise grad_accum_steps to keep "
            "the effective batch), enable amp, or set freeze_backbone, then restart."
        )

    def _train_step(self, pixel_values: torch.Tensor, labels: torch.Tensor) -> float:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        try:
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = compute_logits_for_batch(self.encoder, pixel_values)
            # BCE on fp32 logits: numerically safer than under fp16 autocast.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
            if not torch.isfinite(loss):
                raise NonFiniteLossError(
                    f"Non-finite training loss at epoch={self.state.epoch}, "
                    f"optimizer_step={self.state.global_step}: {loss.item()!r}. Stopping before backward/"
                    "optimizer step; no checkpoint was written for this epoch."
                )
            self.scaler.scale(loss / self.config.grad_accum_steps).backward()
        except torch.cuda.OutOfMemoryError as exc:
            raise self._oom_error(exc, pixel_values.shape[0], "forward/backward") from exc
        return float(loss.item())

    def _optimizer_step(self) -> None:
        try:
            if self.config.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.grad_clip_norm)
            # GradScaler SKIPS optimizer.step() when it finds inf/nan grads
            # (common on the first AMP steps while the scale calibrates) and
            # then shrinks the scale. Only advance the LR schedule when the
            # parameters were really updated.
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            else:
                self.state.skipped_optimizer_steps += 1
            self.optimizer.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError as exc:
            raise self._oom_error(exc, self.config.batch_size, "optimizer step") from exc

    def _reseed_loader_for_epoch(self, loader: DataLoader, epoch: int) -> None:
        """Data order for epoch E depends only on (seed, E), so a resumed
        run reproduces the uninterrupted run's batches exactly."""
        epoch_seed = self.config.seed + epoch
        sampler_generator = getattr(loader.sampler, "generator", None)
        if isinstance(sampler_generator, torch.Generator):
            sampler_generator.manual_seed(epoch_seed)
        if isinstance(loader.generator, torch.Generator):
            loader.generator.manual_seed(epoch_seed)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def train_epoch(self) -> tuple[float, float]:
        """Returns (mean training loss, mean per-micro-batch step time in seconds)."""
        self.encoder.train()
        self._reseed_loader_for_epoch(self.train_loader, self.state.epoch)
        total_loss = 0.0
        total_step_time = 0.0
        num_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_index, (pixel_values, labels) in enumerate(self.train_loader):
            self._sync()
            step_start = time.perf_counter()
            loss_value = self._train_step(pixel_values, labels)
            total_loss += loss_value
            num_batches += 1

            if (batch_index + 1) % self.config.grad_accum_steps == 0:
                self._optimizer_step()
                self.state.global_step += 1
                if self.state.global_step % self.config.log_every_n_steps == 0:
                    self.logger.log(
                        LogRecord(
                            epoch=self.state.epoch, step=self.state.global_step, split="train",
                            loss=loss_value, lr=self.scheduler.get_last_lr()[0],
                            gpu_memory_mb=self._peak_allocated_mb(),
                        )
                    )
            self._sync()
            total_step_time += time.perf_counter() - step_start

        if num_batches % self.config.grad_accum_steps != 0:
            self._optimizer_step()  # flush the trailing partial accumulation window
            self.state.global_step += 1

        return total_loss / max(1, num_batches), total_step_time / max(1, num_batches)

    @torch.no_grad()
    def validate(self) -> tuple[ValidationMetrics, float] | None:
        """Returns (metrics, mean validation BCE loss), or None without a validation set."""
        if self.val_loader is None:
            return None
        self.encoder.eval()
        all_labels: list[float] = []
        all_scores: list[float] = []
        total_loss = 0.0
        for pixel_values, labels in self.val_loader:
            pixel_values = pixel_values.to(self.device)
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = compute_logits_for_batch(self.encoder, pixel_values)
            logits = logits.float().cpu()
            total_loss += float(
                torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
            )
            all_scores.extend(torch.sigmoid(logits).numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
        metrics = compute_validation_metrics(np.array(all_labels), np.array(all_scores))
        return metrics, total_loss / max(1, len(all_labels))

    def _selection_value(
        self, metrics: ValidationMetrics | None, val_loss: float | None, train_loss: float
    ) -> tuple[float, str]:
        """(value, metric_name) where HIGHER is always better (task 16):
        1. validation AUROC, when selection_metric == "val_auroc" and it's defined;
        2. else negative validation loss (AUROC undefined: the validation
           split lacks one class, so ranking quality can't be measured);
        3. else negative training loss (no validation set at all).
        The metric can't flip between epochs: whether AUROC is defined
        depends only on the (fixed) validation labels. Exact ties (AUROC
        saturates at 1.0) are broken by lower validation loss in fit()."""
        if metrics is not None and self.config.selection_metric == "val_auroc" and metrics.threshold_free.auroc is not None:
            return metrics.threshold_free.auroc, "val_auroc"
        if val_loss is not None:
            return -val_loss, "neg_val_loss"
        return -train_loss, "neg_train_loss"

    def _peak_allocated_mb(self) -> float | None:
        return torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else None

    def fit(self, resume_from: str | Path | None = None, stop_after_epochs: int | None = None) -> list[EpochResult]:
        """Runs up to config.epochs epochs (fewer on early stopping).
        `stop_after_epochs` ends *this invocation* after that many epochs,
        exactly as an interruption between epochs would - used to verify
        resume; it is not a config setting."""
        if resume_from is not None:
            self.resume(resume_from)

        results: list[EpochResult] = []
        epochs_this_run = 0

        while self.state.epoch < self.config.epochs:
            if (
                self.config.early_stopping_patience is not None
                and self.state.epochs_without_improvement >= self.config.early_stopping_patience
            ):
                break  # also stops a resume of an already-early-stopped run
            if stop_after_epochs is not None and epochs_this_run >= stop_after_epochs:
                break

            epoch = self.state.epoch
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_start = time.perf_counter()

            train_loss, mean_step_time = self.train_epoch()
            train_time = time.perf_counter() - epoch_start

            val_start = time.perf_counter()
            validation = self.validate()
            self._sync()
            val_time = time.perf_counter() - val_start
            val_metrics, val_loss = validation if validation is not None else (None, None)
            epoch_time = time.perf_counter() - epoch_start

            peak_allocated = self._peak_allocated_mb()
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / 2**20 if self.device.type == "cuda" else None

            selection_value, metric_name = self._selection_value(val_metrics, val_loss, train_loss)
            tiebreak = -val_loss if val_loss is not None else -train_loss
            is_best = (
                self.state.best_metric is None
                or selection_value > self.state.best_metric
                or (
                    selection_value == self.state.best_metric
                    and self.state.best_tiebreak is not None
                    and tiebreak > self.state.best_tiebreak
                )
            )
            if is_best:
                self.state.best_metric = selection_value
                self.state.best_tiebreak = tiebreak
                self.state.best_metric_name = metric_name
                self.state.best_epoch = epoch
                self.state.epochs_without_improvement = 0
            else:
                self.state.epochs_without_improvement += 1
            self.state.epoch = epoch + 1

            metrics_dict: dict[str, Any] = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection_metric": metric_name,
                "selection_value": selection_value,
                "is_best": is_best,
                "optimizer_steps_attempted": self.state.global_step,
                "optimizer_steps_skipped_amp": self.state.skipped_optimizer_steps,
                "amp_loss_scale": self.scaler.get_scale() if self.use_amp else None,
                "train_time_seconds": train_time,
                "val_time_seconds": val_time,
                "mean_step_time_seconds": mean_step_time,
                "peak_gpu_reserved_mb": peak_reserved,
                "val": val_metrics.to_dict() if val_metrics is not None else None,
            }
            self.logger.log(
                LogRecord(
                    epoch=epoch, step=self.state.global_step, split="epoch",
                    loss=train_loss, lr=self.scheduler.get_last_lr()[0],
                    epoch_time_seconds=epoch_time, gpu_memory_mb=peak_allocated,
                    metrics=metrics_dict,
                )
            )
            results.append(
                EpochResult(
                    epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_metrics=val_metrics,
                    epoch_time_seconds=epoch_time, train_time_seconds=train_time, val_time_seconds=val_time,
                    mean_step_time_seconds=mean_step_time, peak_gpu_memory_mb=peak_allocated,
                    peak_gpu_reserved_mb=peak_reserved, selection_metric_name=metric_name,
                    selection_value=selection_value, is_best=is_best,
                )
            )

            # "latest" is saved AFTER the state update so resuming from it
            # continues with the next epoch and the right early-stop counter.
            self.save("latest", metrics_dict)
            if is_best:
                self.save("best", metrics_dict)
            epochs_this_run += 1

        return results


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    encoder: DeepfakeVisualEncoder,
    val_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> ValidationMetrics:
    """Evaluation-only path (task 19): loads just the model weights from a
    checkpoint and runs validation - no optimizer/scheduler needed. The
    encoder must match the checkpoint's recorded encoder_name."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    if checkpoint.provenance.encoder_name != encoder.spec.name:
        raise ValueError(
            f"Checkpoint was trained with encoder {checkpoint.provenance.encoder_name!r}, "
            f"but evaluation was asked to use {encoder.spec.name!r}."
        )
    encoder = encoder.to(device)
    encoder.load_state_dict(checkpoint.model_state)
    encoder.eval()

    all_labels: list[float] = []
    all_scores: list[float] = []
    for pixel_values, labels in val_loader:
        pixel_values = pixel_values.to(device)
        logits = compute_logits_for_batch(encoder, pixel_values)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    return compute_validation_metrics(np.array(all_labels), np.array(all_scores), threshold=threshold)
