"""Reproducible training pipeline for the Phase 4 encoders.

Everything here operates on synthetic fixtures and the already-authorized
MobileNetV4/EfficientNet-B0 weights - no dataset, GenD, DINOv2, or
additional pretrained model is downloaded by this subpackage. Any
"training" run against real deepfake data still requires the user to
provide a manifest (configuard.datasets, Phase 3) built from their own
locally-obtained dataset. Results produced against the synthetic fixtures
in tests/training/ are engineering smoke tests, not deepfake-detection
accuracy claims - see docs/MODEL_CARD.md.
"""
