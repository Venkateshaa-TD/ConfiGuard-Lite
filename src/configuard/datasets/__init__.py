"""Dataset registry, canonical sample schema, adapters, leakage-safe
splitting, duplicate detection, and storage-path checking.

Nothing in this subpackage downloads data. Adapters for access-controlled
datasets (FaceForensics++, Celeb-DF-v2, DFDC, DF40, DeeperForensics-1.0)
scan a user-provided local root and fail clearly (DatasetAccessError) if
it is missing - see docs/DATASETS.md for each dataset's official access
process and this phase's "verification required" notes on folder
structure assumptions that could not be checked against real data.
"""
