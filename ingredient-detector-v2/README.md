# Ingredient Detector v2

YOLO-based multi-object detection for raw ingredients in varied scenes. The target
evaluation is per-image ingredient-set F1, overall and by `scene_type`.

## Setup

Use Python 3.10. Install the verified CUDA 12.1 pair, then the project requirements:

```powershell
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121
python -m pip install -r requirements.txt
```

Keep `ROBOFLOW_API_KEY` in the environment when Roboflow access is needed; never put it
in source files.

`configs/classes.yaml` is the class taxonomy source of truth. Generate
`configs/data.yaml` with `python -m src.data.config_utils`; the generated dataset root
is an absolute path.

## Data and evaluation

Do not train or tune on `data/stress_test` or `data/labeled/test`. Split personal photos
by session and public photos by near-duplicate cluster. Public data downloads require
prior review of source, expected size, licence, and available disk space.

Evaluate predictions as ingredient sets with
[`src/eval/set_metrics.py`](./src/eval/set_metrics.py). Overall scores are macro-averaged
per image; class scores are one-vs-rest micro metrics; scene scores are macro-averaged
per image within each `scene_type`. Empty truth and prediction sets score 1; when only one
side is empty, the image scores 0. Keep threshold selection on validation data only.

## Training

The 50-image overfit sanity check completed. Stage A was stopped after 45 of 100 epochs;
its latest checkpoint is resumable but is not a final model. See
[`RESULTS.md`](./RESULTS.md) for measured public-validation scores. To continue Stage A,
run this from the repository root:

```powershell
.\.venv\Scripts\yolo.exe detect train model=runs/train/stage-a-yolov8s/weights/last.pt resume=True workers=4 device=0
```

The overfit sanity check used 50 images only from `data/public/train/images` with seed 42.
For reproducibility, the command below rebuilds `runs/overfit50/data.yaml` with both
`train` and `val` referencing the same 50-image list and class names from
`data/public/data.yaml`. This intentional reuse checks only that the model can fit labels;
its metrics are diagnostic, not validation results.

```powershell
@'
from pathlib import Path
import random
import yaml

root = Path.cwd()
data = yaml.safe_load((root / "data/public/data.yaml").read_text(encoding="utf-8"))
images = sorted((root / "data/public/train/images").glob("*"))
selected = random.Random(42).sample(images, 50)
out = root / "runs/overfit50"
out.mkdir(parents=True, exist_ok=True)
image_list = out / "images.txt"
image_list.write_text(
    "\n".join(str(path.resolve()) for path in selected) + "\n",
    encoding="utf-8",
)
data["path"] = str((root / "data/public").resolve())
data["train"] = str(image_list.resolve())
data["val"] = str(image_list.resolve())
(out / "data.yaml").write_text(
    yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
)
'@ | .\.venv\Scripts\python.exe -

yolo detect train model=yolov8s.pt data=runs/overfit50/data.yaml epochs=100 imgsz=640 batch=16 device=0 workers=0 seed=42 project=runs/train name=overfit50
```

Check that training can fit the 50-image diagnostic subset before proceeding; do not use
this run to tune thresholds or report generalization. If CUDA memory is insufficient,
retry with `batch=8`.

Stage A trains the main YOLOv8s model on the merged public train split and uses only its
public validation split for evaluation:

```powershell
yolo detect train model=yolov8s.pt data=data/public/data.yaml epochs=100 imgsz=640 batch=16 device=0 workers=0 seed=42 project=runs/train name=stage-a-yolov8s
```

Stage B requires my own labeled photos in `data/labeled`; do not use them until ready.
Generate the personal-data YAML, fine-tune with the backbone frozen for 10 epochs at
`lr0=3e-4`, then initialize from that run's best checkpoint and unfreeze. The second
10-epoch block is an initial plan; use validation-only monitoring for any further tuning.

```powershell
python -m src.data.config_utils
yolo detect train model=runs/train/stage-a-yolov8s/weights/best.pt data=configs/data.yaml epochs=10 imgsz=640 batch=16 device=0 workers=0 seed=42 lr0=0.0003 freeze=10 project=runs/train name=stage-b-frozen
yolo detect train model=runs/train/stage-b-frozen/weights/best.pt data=configs/data.yaml epochs=10 imgsz=640 batch=16 device=0 workers=0 seed=42 lr0=0.0003 freeze=0 project=runs/train name=stage-b-unfrozen
```

Do not run final test evaluation until the final model is selected; use
`data/labeled/test` once per final model. Never train or tune on
`data/labeled/test` or `data/stress_test`.

## Integrating a trained model

The detector checkpoint is a YOLO `.pt` file. Keep model weights out of the source-code
repository (`models/` and `runs/` are ignored by Git). Share a finished checkpoint as a
GitHub Release asset, then download it into `models/`, for example
`models/ingredient-detector-best.pt`. Keep the model's training report and applicable
dataset/model license notices with the release.

Preview model and source bundle: [v0.1.0-preview release](https://github.com/himanshuT841/dl-cnn-model/releases/tag/v0.1.0-preview).
The `ingredient-detector-v2-full-preview-bundle.zip` asset contains the project source,
integration code, taxonomy, and checkpoint. The separate `.pt` asset is the best checkpoint
from epoch 43 of an interrupted 100-epoch run, not a finished model. Neither asset contains
training datasets, personal photos, generated machine-specific `configs/data.yaml`, or
virtual-environment files. After extracting the bundle, install requirements, add or prepare
the dataset locally if needed, and set the model path as shown below.

Set the checkpoint path and start the local FastAPI inference page from PowerShell:

```powershell
$env:INGREDIENT_MODEL_PATH = (Resolve-Path ".\models\ingredient-detector-best.pt").Path
.\.venv\Scripts\python.exe -m src.infer.app
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload a JPEG, PNG, WebP, or BMP
image up to 10 MB, choose a confidence threshold, and view ingredient detections,
approximate counts, and the annotated image. The environment variable applies to that
PowerShell session; set it again in a new terminal.

Without an explicit path, the app loads a completed Stage B checkpoint, then a completed
Stage A checkpoint. It checks recorded epoch counts so interrupted checkpoints are not
mistaken for completed training. If neither exists, it uses `models/yolov8s.pt` and labels
the page **demo only**; these are generic COCO-pretrained weights, not the ingredient
detector. An explicit checkpoint from an incomplete Stage A or Stage B run is labeled
**preview only**. The service binds to localhost and is intended for local testing.

## Licensing

Ultralytics YOLO is licensed under AGPL-3.0. Review its terms before distributing or
deploying this project.
