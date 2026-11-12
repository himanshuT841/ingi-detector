# Project

Final year project "Image-Based Recipe Suggestion". Build the ingredient-detection CNN:
given one photo of RAW ingredients in ANY scene (fridge, counter, market stall, basket or
bag, grocery checkout), return the SET of ingredients with confidence and approximate
count. This is multi-object detection. Key metric: per-image ingredient-set F1, overall and
per scene_type.

# Environment

Windows PowerShell, Python 3.10 already installed, RTX 4060 8 GB (batch 16 at 640px, fall
back to 8). PyTorch (CUDA 12.1 build), Ultralytics YOLO (YOLOv8s main; YOLOv8n and YOLO11s
for comparison), OpenCV, FastAPI, pytest, imagehash, FiftyOne, roboflow. Ultralytics is
AGPL-3.0: mention in README. configs/classes.yaml is the single source of truth; data.yaml
is generated from it with an ABSOLUTE path. Training scripts need
if __name__ == "__main__": on Windows. Final target: ONNX on PC, TensorRT FP16 engine built
ON the Jetson (never copy an engine from the PC).

# Hard rules

- NEVER invent metrics, dataset sizes or benchmark numbers. Run things and report real output.
- data/stress_test and data/labeled/test will contain ONLY my own photos. Never train or
  tune on them. Tune only on val. Use test once per final model.
- Use ROBOFLOW_API_KEY from the environment; never hardcode keys.
- Before any download show: source, expected size, licence and free disk space, then wait for
  my approval. Total public data budget: 60 GB unless I approve more.
- Split my own photos by SESSION (SESSION_idx.jpg). Split public data by near-duplicate
  CLUSTER, never separating near-copies across splits.
- No synthetic datasets for validation or test.
- Augmented Roboflow exports contain copies of the same photo: count UNIQUE images, and
  prefer versions without augmentation.
- Keep code short, typed, commented where non-obvious, seeds fixed (seed 42).
- Finish one STEP, report real outputs, and stop. Do not start the next step.
