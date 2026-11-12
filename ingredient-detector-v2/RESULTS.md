# Results

Record each run's command, seed, Git commit, and measured metrics here; do not estimate or
invent results.

| Run | Command | Seed | Git commit | Measured metrics |
|---|---|---:|---|---|
| Overfit sanity (50 public train images) | `yolo detect train model=models\yolov8s.pt data=runs\overfit50\data.yaml epochs=100 imgsz=640 batch=16 device=0 workers=0 seed=42 project=runs\train name=overfit50` | 42 | Unavailable; Git is not installed in the current environment | Completed 100 epochs. On the same 50-image diagnostic subset: P=0.971, R=0.952, mAP50=0.979, mAP50-95=0.927. Diagnostic only, not a generalization score. Checkpoint: `runs\detect\runs\train\overfit50\weights\best.pt`. |
| Stage A: YOLOv8s public data | `yolo detect train model=runs/train/stage-a-yolov8s/weights/last.pt resume=True workers=4 device=0` (resumes the original 100-epoch run) | 42 | Unavailable; Git is not installed in the current environment | Stopped after 45/100 epochs. Latest public-val P=0.633, R=0.613, mAP50=0.636, mAP50-95=0.463 (epoch 45); best recorded mAP50=0.637 (epoch 43). Checkpoint is resumable, not final. |
| Stage B: frozen then unfrozen personal-data fine-tune | Not run | 42 (planned) | Unavailable; Git is not installed in the current environment | Not measured; waiting for labeled personal photos. |

Public-data audit counts before training: 39 of 97 configured classes have instances.
`cabbage` has 783 train and 179 val instances; `cauliflower` has 221 train and 46 val
instances. Classes with no labeled examples cannot be learned by this run.

## Stage A checkpoint preview (public validation only)

Measured on `data/public/val` with checkpoint snapshot `runs/train/stage-a-yolov8s/weights/preview-epoch-35.pt`.
No files from `data/labeled/test` or `data/stress_test` were read. These validation scores
are not held-out test results, and the checkpoint is from an incomplete run.

| Metric | Result |
|---|---:|
| Validation images | 5,108 |
| Box precision | 0.635 |
| Box recall | 0.595 |
| Box mAP50 | 0.629 |
| Box mAP50-95 | 0.457 |
| Macro per-image ingredient-set precision (confidence 0.25) | 0.8841 |
| Macro per-image ingredient-set recall (confidence 0.25) | 0.9303 |
| Macro per-image ingredient-set F1 (confidence 0.25) | 0.8954 |
| Exact ingredient-set match (confidence 0.25) | 0.8107 |
| Cabbage box mAP50 | 0.499 |
| Cauliflower box mAP50 | 0.544 |
| Cabbage per-image set F1 (support: 67 images) | 0.7417 |
| Cauliflower per-image set F1 (support: 24 images) | 0.7273 |

Per-image set scores treat each image as a set of ingredient names (multiple detections of
the same ingredient in one image count once). The public validation export does not contain
verified scene-type metadata, so no per-scene score is reported.

The best checkpoint (`best.pt`, best epoch 43) was published as the explicitly incomplete
public pre-release
[`v0.1.0-preview`](https://github.com/himanshuT841/dl-cnn-model/releases/tag/v0.1.0-preview).
Asset: `ingredient-detector-v2-stage-a-preview-epoch-43.pt` (67,329,592 bytes).
SHA-256: `b209e8a99a65adec26bf6bb8a71516d3713bf8da35b97805a9eed6ec6f3ec254`.

The same release contains
[`ingredient-detector-v2-full-preview-bundle.zip`](https://github.com/himanshuT841/dl-cnn-model/releases/download/v0.1.0-preview/ingredient-detector-v2-full-preview-bundle.zip)
(62,062,883 bytes). It contains the project source, tests, taxonomy, docs, and the same
checkpoint; it excludes datasets, user photos, virtual environments, run outputs, and the
generated absolute-path `configs/data.yaml`. Bundle SHA-256:
`2e01fbe5f4f69e906fa7b98ce62d5faaa49f6799d79a026e064dc3bee1a0da10`.
