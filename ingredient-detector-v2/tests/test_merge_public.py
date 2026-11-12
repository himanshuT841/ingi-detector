from pathlib import Path
from types import SimpleNamespace

import yaml
import pytest

from src.data.merge_public import (
    _get_roboflow_version,
    _cluster_sample_hashes,
    _suppress_box_polygon_duplicates,
    inspect_yolo_export,
    load_taxonomy,
    map_source_class,
    run_dry_run,
)


def test_food_aliases_map_and_nonfood_is_not_promoted() -> None:
    taxonomy_path = Path(__file__).parents[1] / "configs" / "classes.yaml"
    _, aliases = load_taxonomy(taxonomy_path)

    assert map_source_class("Bell Pepper", aliases) == "capsicum"
    assert map_source_class("Unlisted food", aliases) == "other_food"
    assert map_source_class("ham", aliases) == "other_food"
    assert map_source_class("squash/patisson", aliases) == "squash"
    assert map_source_class("person", aliases) is None
    assert map_source_class("water bottle", aliases) is None


def test_inspect_export_counts_only_images_with_retained_boxes(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        "classes:\n"
        "  - name: tomato\n"
        "    synonyms: [tomatoes]\n"
        "  - name: other_food\n"
        "    synonyms: []\n",
        encoding="utf-8",
    )
    export_dir = tmp_path / "export"
    (export_dir / "images" / "train").mkdir(parents=True)
    (export_dir / "labels" / "train").mkdir(parents=True)
    (export_dir / "data.yaml").write_text(
        yaml.safe_dump({"names": ["tomatoes", "bread", "person"]}),
        encoding="utf-8",
    )
    (export_dir / "images" / "train" / "one.jpg").write_bytes(b"")
    (export_dir / "images" / "train" / "two.jpg").write_bytes(b"")
    (export_dir / "labels" / "train" / "one.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
        encoding="utf-8",
    )
    (export_dir / "labels" / "train" / "two.txt").write_text(
        "2 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    names, aliases = load_taxonomy(taxonomy_path)

    report = inspect_yolo_export(export_dir, names, aliases)

    assert report.total_images == 2
    assert report.kept_images == 1
    assert report.instances_per_class == {"tomato": 1, "other_food": 1}
    assert report.top_unmapped == [("person", 1)]


def test_inspect_export_supports_roboflow_split_directories(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        "classes:\n  - name: tomato\n    synonyms: []\n",
        encoding="utf-8",
    )
    export_dir = tmp_path / "export"
    images_dir = export_dir / "train" / "images"
    labels_dir = export_dir / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (export_dir / "data.yaml").write_text(
        yaml.safe_dump({"names": ["tomato"]}),
        encoding="utf-8",
    )
    (images_dir / "one.jpg").write_bytes(b"")
    (labels_dir / "one.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    names, aliases = load_taxonomy(taxonomy_path)

    report = inspect_yolo_export(export_dir, names, aliases)

    assert report.total_images == 1
    assert report.kept_images == 1
    assert report.instances_per_class == {"tomato": 1}


def test_near_duplicate_hashes_remain_in_one_cluster() -> None:
    clusters = _cluster_sample_hashes(
        [0x1234567890ABCDEF, 0x1234567890ABCDEE, 0xFFFFFFFFFFFFFFFF]
    )

    assert sorted(sorted(cluster) for cluster in clusters) == [[0, 1], [2]]


def test_box_inside_same_class_polygon_is_suppressed() -> None:
    boxes = [(0, 0.5, 0.5, 0.8, 0.8), (1, 0.5, 0.5, 0.8, 0.8)]
    polygons = [(0, 0.5, 0.5, 0.4, 0.4)]

    retained, suppressed = _suppress_box_polygon_duplicates(boxes, polygons)

    assert retained == [boxes[1]]
    assert suppressed == 1


def test_read_merge_samples_converts_polygons_and_drops_duplicate_box(
    tmp_path: Path, monkeypatch
) -> None:
    import src.data.merge_public as merge_public
    from PIL import Image

    export_dir = tmp_path / "export"
    images_dir = export_dir / "train" / "images"
    labels_dir = export_dir / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (export_dir / "data.yaml").write_text(
        yaml.safe_dump({"names": ["tomato"]}),
        encoding="utf-8",
    )
    Image.new("RGB", (16, 16), "red").save(images_dir / "one.jpg")
    (labels_dir / "one.txt").write_text(
        "0 0.5 0.5 0.8 0.8\n"
        "0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(merge_public, "_local_export", lambda _: export_dir)

    samples, dropped, flags, polygon_rows, suppressed = (
        merge_public._read_merge_samples(
            {"workspace": "sample", "project": "tomatoes", "version": 1},
            ["tomato"],
            {"tomato": "tomato"},
        )
    )

    assert len(samples) == 1
    assert samples[0].boxes[0] == pytest.approx((0, 0.5, 0.5, 0.4, 0.4))
    assert not dropped
    assert not flags
    assert polygon_rows == 1
    assert suppressed == 1


def test_local_dry_run_does_not_create_or_copy_data(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import src.data.merge_public as merge_public

    monkeypatch.setattr(merge_public, "PROJECT_ROOT", tmp_path)
    names, aliases = load_taxonomy(
        Path(__file__).parents[1] / "configs" / "classes.yaml"
    )
    sources = [
        {
            "workspace": "sample-workspace",
            "project": "sample-project",
            "version": 1,
            "scene_type": "to verify",
        }
    ]

    run_dry_run(sources, names, aliases, api_key=None, local_only=True)

    output = capsys.readouterr().out
    assert "local-only dry run" in output
    assert "total images: unavailable" in output
    assert not (tmp_path / "data").exists()


def test_unpinned_roboflow_project_lists_versions_without_selecting_one(
    capsys,
) -> None:
    version = SimpleNamespace(version="4", images=125)
    project = SimpleNamespace(classes=["tomato"], versions=lambda: [version])
    rf = SimpleNamespace(
        workspace=lambda _: SimpleNamespace(project=lambda _: project)
    )

    _, selected, classes = _get_roboflow_version(
        rf,
        {"workspace": "sample", "project": "produce", "version": None},
    )

    assert selected is None
    assert classes == ["tomato"]
    assert "available versions (not selected): v4 (125 images)" in capsys.readouterr().out


def test_missing_requested_version_is_not_replaced_by_another_version(
    capsys,
) -> None:
    available = SimpleNamespace(version="2", images=80)

    def missing_version(_: int) -> None:
        raise RuntimeError("Version number 1 is not found.")

    project = SimpleNamespace(
        classes=["tomato"],
        version=missing_version,
        versions=lambda: [available],
    )
    rf = SimpleNamespace(
        workspace=lambda _: SimpleNamespace(project=lambda _: project)
    )

    _, selected, _ = _get_roboflow_version(
        rf,
        {"workspace": "sample", "project": "produce", "version": 1},
    )

    output = capsys.readouterr().out
    assert selected is None
    assert "requested version unavailable" in output
    assert "v2 (80 images)" in output
