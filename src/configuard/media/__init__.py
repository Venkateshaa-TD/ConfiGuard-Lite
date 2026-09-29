"""Face and media preprocessing: decoding, frame sampling, face detection,
tracking, alignment, and a versioned face-crop cache.

This subpackage is deliberately independent of configuard.pipeline (the
Phase 1 dummy end-to-end slice). Wiring real face crops into the model
pipeline happens once a real encoder exists (see docs/PROJECT_PLAN.md).
"""
