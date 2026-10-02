// The ONLY model statistics shown on the site. Each is recorded in docs/EXPERIMENT_LOG.md and is a
// FaceForensics++ (c23) official-validation-split or artifact fact, not a claim about other data.
export const DATASET_LABEL = "FaceForensics++ c23 · official validation split (695 videos)";

export const FACTS = [
  { value: "2.49M", unit: "parameters", label: "MobileNetV4-Conv-Small student (2,494,305 parameters)", scope: "model size" },
  { value: "9.5", unit: "MiB", label: "ONNX FP32 model file (9.49 MiB) used for CPU inference", scope: "artifact size" },
  { value: "6.19", unit: "frames", label: "Average frames analysed per video by adaptive 4/8/16 sampling", scope: DATASET_LABEL },
  { value: "0.9735", unit: "AUROC", label: "Video-level AUROC, 16-frame mean score", scope: DATASET_LABEL },
] as const;

export const SCOPE_NOTE =
  "Validation results on FaceForensics++ face manipulations (Deepfakes, Face2Face, FaceSwap, NeuralTextures). " +
  "They do not establish performance on other datasets, generators or real-world media.";
