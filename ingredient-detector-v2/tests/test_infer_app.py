from pathlib import Path

from src.infer.app import _choose_model_path


def _create_complete_run(run_dir: Path, epochs: int) -> None:
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights/best.pt").touch()
    (run_dir / "args.yaml").write_text(f"epochs: {epochs}\n", encoding="utf-8")
    rows = ["epoch"]
    rows.extend(str(epoch) for epoch in range(epochs))
    (run_dir / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_choose_model_prefers_latest_trained_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    import src.infer.app as infer_app

    stage_a_run = tmp_path / "runs/train/stage-a-yolov8s"
    stage_b_run = tmp_path / "runs/train/stage-b-unfrozen"
    _create_complete_run(stage_a_run, 100)
    _create_complete_run(stage_b_run, 10)
    monkeypatch.setattr(infer_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("INGREDIENT_MODEL_PATH", raising=False)

    path, kind = _choose_model_path()

    assert path == stage_b_run / "weights/best.pt"
    assert kind == "trained"


def test_choose_model_skips_incomplete_run(
    tmp_path: Path, monkeypatch
) -> None:
    import src.infer.app as infer_app

    stage_a_run = tmp_path / "runs/train/stage-a-yolov8s"
    _create_complete_run(stage_a_run, 100)
    (stage_a_run / "results.csv").write_text("epoch\n0\n1\n2\n3\n", encoding="utf-8")
    demo_weights = tmp_path / "models/yolov8s.pt"
    demo_weights.parent.mkdir(parents=True)
    demo_weights.touch()
    monkeypatch.setattr(infer_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("INGREDIENT_MODEL_PATH", raising=False)

    path, kind = _choose_model_path()

    assert path == demo_weights
    assert kind == "pretrained-demo"


def test_configured_checkpoint_in_incomplete_run_is_marked_preview(
    tmp_path: Path, monkeypatch
) -> None:
    import src.infer.app as infer_app

    run_dir = tmp_path / "runs/train/stage-a-yolov8s"
    _create_complete_run(run_dir, 100)
    (run_dir / "results.csv").write_text("epoch\n0\n1\n", encoding="utf-8")
    preview = run_dir / "weights/preview.pt"
    preview.touch()
    monkeypatch.setattr(infer_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("INGREDIENT_MODEL_PATH", str(preview))

    path, kind = _choose_model_path()

    assert path == preview
    assert kind == "trained-preview"


def test_choose_model_falls_back_to_explicitly_marked_demo_weights(
    tmp_path: Path, monkeypatch
) -> None:
    import src.infer.app as infer_app

    demo_weights = tmp_path / "models/yolov8s.pt"
    demo_weights.parent.mkdir(parents=True)
    demo_weights.touch()
    monkeypatch.setattr(infer_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("INGREDIENT_MODEL_PATH", raising=False)

    path, kind = _choose_model_path()

    assert path == demo_weights
    assert kind == "pretrained-demo"


def test_configured_missing_checkpoint_does_not_fall_back_to_demo(
    tmp_path: Path, monkeypatch
) -> None:
    import src.infer.app as infer_app

    demo_weights = tmp_path / "models/yolov8s.pt"
    demo_weights.parent.mkdir(parents=True)
    demo_weights.touch()
    monkeypatch.setattr(infer_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("INGREDIENT_MODEL_PATH", "missing/best.pt")

    path, kind = _choose_model_path()

    assert path is None
    assert kind == "unavailable"
