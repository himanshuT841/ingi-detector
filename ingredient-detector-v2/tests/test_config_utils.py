from pathlib import Path

import yaml

from src.data.config_utils import generate_data_yaml


def test_generate_data_yaml_uses_absolute_dataset_path_and_class_order(
    tmp_path: Path,
) -> None:
    classes_path = tmp_path / "classes.yaml"
    classes_path.write_text(
        "classes:\n  - name: tomato\n  - name: onion\n",
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "my_data"
    output_path = tmp_path / "configs" / "data.yaml"

    result = generate_data_yaml(classes_path, dataset_dir, output_path)
    config = yaml.safe_load(result.read_text(encoding="utf-8"))

    assert result == output_path
    assert config["path"] == str(dataset_dir.resolve())
    assert config["names"] == {0: "tomato", 1: "onion"}
    assert config["nc"] == 2


def test_generate_public_data_yaml_uses_merged_split_layout(tmp_path: Path) -> None:
    classes_path = tmp_path / "classes.yaml"
    classes_path.write_text("classes:\n  - name: tomato\n", encoding="utf-8")
    dataset_dir = tmp_path / "public"

    result = generate_data_yaml(
        classes_path=classes_path,
        dataset_dir=dataset_dir,
        output_path=dataset_dir / "data.yaml",
        train_path="train/images",
        val_path="val/images",
        test_path=None,
    )
    config = yaml.safe_load(result.read_text(encoding="utf-8"))

    assert config["path"] == str(dataset_dir.resolve())
    assert config["train"] == "train/images"
    assert config["val"] == "val/images"
    assert "test" not in config
