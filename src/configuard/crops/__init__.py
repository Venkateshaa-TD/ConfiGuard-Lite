"""Phase 5d: matched, leakage-preserving face-crop extraction.

Modules:
- families:  FF++ content families (target original + its fakes) built
             from the official split manifests.
- matching:  shared-range nested 4/8/16 sampling and deterministic
             nearby-frame recovery (identical rule for every class).
- alignment: five-landmark similarity alignment to a fixed template.
- store:     config-keyed, atomic, stale-refusing crop store.
- extract:   per-family extraction orchestration.
- manifests: frame-level crop manifests, matched pairs, audit sidecar,
             leakage re-validation and detection/recovery statistics.
"""
