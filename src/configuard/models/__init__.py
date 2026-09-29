"""Pretrained visual-encoder backbones (MobileNetV4-Conv-Small,
EfficientNet-B0), ONNX export/verification, and image/fixed-frame video
inference.

IMPORTANT: every prediction from configuard.models is from an
ImageNet-pretrained backbone with a freshly, randomly initialized binary
head - it has NEVER been trained on deepfake data. Nothing here should be
read as a meaningful "real"/"fake" judgment - see
DeepfakeVisualEncoder.PREDICTION_DISCLAIMER and docs/MODEL_CARD.md.

Importing this package does not itself trigger any download; weights are
only fetched the first time a specific pretrained model is constructed.
Set HF_HOME/HF_HUB_CACHE/TORCH_HOME (via configuard.env_loader.load_dotenv,
called BEFORE this import) to control where those weights are cached -
see docs/DECISIONS.md.
"""
