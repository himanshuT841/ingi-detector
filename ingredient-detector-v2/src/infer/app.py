"""Local browser demo and API for ingredient detection."""

import base64
import csv
import os
from collections import Counter
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote

import cv2
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from ultralytics import YOLO

from src.data.merge_public import load_taxonomy, map_source_class

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
_model: Any = None
_model_path: Path | None = None
_model_kind = "unavailable"
_aliases: dict[str, str] = {}


class IngredientCount(BaseModel):
    ingredient: str
    count: int
    confidence: float


class PredictionResponse(BaseModel):
    filename: str
    model: str
    model_kind: str
    warning: str | None
    ingredients: list[IngredientCount]
    annotated_image: str


def _completed_checkpoint(run_dir: Path) -> Path | None:
    checkpoint = run_dir / "weights/best.pt"
    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"
    if not all(path.is_file() for path in (checkpoint, args_path, results_path)):
        return None

    run_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(run_args, dict) or not isinstance(run_args.get("epochs"), int):
        raise ValueError(f"Invalid training arguments in {args_path}")
    with results_path.open(newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))
    if not rows:
        return None
    completed_epochs = int(rows[-1]["epoch"]) + 1
    return checkpoint if completed_epochs >= run_args["epochs"] else None


def _choose_model_path() -> tuple[Path | None, str]:
    configured = os.environ.get("INGREDIENT_MODEL_PATH")
    if configured:
        resolved = Path(configured)
        resolved = resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
        if not resolved.is_file():
            return None, "unavailable"
        for run_name in ("stage-b-unfrozen", "stage-a-yolov8s"):
            run_dir = (PROJECT_ROOT / "runs/train" / run_name).resolve()
            try:
                resolved.relative_to(run_dir)
            except ValueError:
                continue
            return resolved, (
                "trained" if _completed_checkpoint(run_dir) is not None else "trained-preview"
            )
        return resolved, "trained"

    for run_name in ("stage-b-unfrozen", "stage-a-yolov8s"):
        checkpoint = _completed_checkpoint(PROJECT_ROOT / "runs/train" / run_name)
        if checkpoint is not None:
            return checkpoint, "trained"
    demo_path = PROJECT_ROOT / "models/yolov8s.pt"
    if demo_path.is_file():
        return demo_path, "pretrained-demo"
    return None, "unavailable"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _aliases, _model, _model_kind, _model_path
    _, _aliases = load_taxonomy()
    _model_path, _model_kind = _choose_model_path()
    _model = YOLO(str(_model_path)) if _model_path is not None else None
    yield
    _model = None


app = FastAPI(title="Ingredient Detector", lifespan=lifespan)


def _model_status() -> dict[str, str | bool]:
    warning = None
    if _model_kind == "pretrained-demo":
        warning = (
            "Demo only: this is the generic COCO-pretrained YOLOv8s model, not "
            "trained on the ingredient dataset."
        )
    elif _model_kind == "trained-preview":
        warning = (
            "Preview only: this checkpoint is from an incomplete training run. "
            "Validation scores are not held-out test results."
        )
    return {
        "available": _model is not None,
        "model": _model_path.name if _model_path else "",
        "model_kind": _model_kind,
        "warning": warning,
    }


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return (PROJECT_ROOT / "src/infer/index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health() -> dict[str, str | bool]:
    return _model_status()


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    conf: Annotated[float, Query(ge=0.01, le=1.0)] = 0.25,
) -> PredictionResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No model checkpoint found. Set INGREDIENT_MODEL_PATH to a trained checkpoint.",
        )
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413, detail="Image exceeds the 10 MB limit."
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid content length."
            ) from exc
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Upload an image as the request body.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            if source_image.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Use a JPEG, PNG, WebP, or BMP image.",
                )
            if source_image.width * source_image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the 40 megapixel limit.",
                )
            image = ImageOps.exif_transpose(source_image).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="The uploaded file is not a valid image."
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400, detail="The uploaded image could not be decoded."
        ) from exc

    result = _model.predict(image, conf=conf, imgsz=640, verbose=False)[0]
    counts: Counter[str] = Counter()
    confidences: dict[str, list[float]] = {}
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            source_name = _model.names[class_id]
            ingredient = map_source_class(str(source_name), _aliases)
            if ingredient is None or ingredient == "other_food":
                continue
            counts[ingredient] += 1
            confidences.setdefault(ingredient, []).append(float(box.conf.item()))

    annotated = result.plot()
    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise RuntimeError("Could not encode the annotated prediction image.")
    kind_warning = _model_status()["warning"]
    return PredictionResponse(
        filename=unquote(request.headers.get("x-filename", "upload")),
        model=_model_path.name if _model_path else "unknown",
        model_kind=_model_kind,
        warning=kind_warning,
        ingredients=[
            IngredientCount(
                ingredient=name,
                count=count,
                confidence=sum(confidences[name]) / count,
            )
            for name, count in sorted(counts.items())
        ],
        annotated_image="data:image/jpeg;base64,"
        + base64.b64encode(encoded.tobytes()).decode("ascii"),
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.infer.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
