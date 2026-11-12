"""Metrics for per-image ingredient sets and their classes and scenes."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AggregateMetrics:
    precision: float
    recall: float
    f1: float
    exact_match: float
    image_count: int


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    support: int


@dataclass(frozen=True)
class SetMetrics:
    overall: AggregateMetrics
    per_class: dict[str, ClassMetrics]
    per_scene_type: dict[str, AggregateMetrics]


def _image_scores(
    truth: set[str], prediction: set[str]
) -> tuple[float, float, float, float]:
    true_positive = len(truth & prediction)
    precision = (
        true_positive / len(prediction) if prediction else float(not truth)
    )
    recall = true_positive / len(truth) if truth else float(not prediction)
    f1_denominator = len(truth) + len(prediction)
    f1 = (
        2 * true_positive / f1_denominator
        if f1_denominator
        else 1.0
    )
    return precision, recall, f1, float(truth == prediction)


def _aggregate(
    scores: list[tuple[float, float, float, float]],
) -> AggregateMetrics:
    count = len(scores)
    return AggregateMetrics(
        precision=sum(score[0] for score in scores) / count,
        recall=sum(score[1] for score in scores) / count,
        f1=sum(score[2] for score in scores) / count,
        exact_match=sum(score[3] for score in scores) / count,
        image_count=count,
    )


def evaluate_sets(
    ground_truth: Iterable[Iterable[str]],
    predictions: Iterable[Iterable[str]],
    scene_types: Iterable[str] | None = None,
    class_names: Iterable[str] | None = None,
) -> SetMetrics:
    """Return macro image-set scores and one-vs-rest class and scene scores.

    An image with empty truth and prediction scores 1; an empty set on only one
    side scores 0. Per-class scores are micro-aggregated across images.
    """
    raw_truth = list(ground_truth)
    raw_predictions = list(predictions)
    if any(isinstance(items, str) for items in raw_truth + raw_predictions):
        raise ValueError("Each image's ingredients must be an iterable of names")
    if any(
        not isinstance(name, str) or not name
        for items in raw_truth + raw_predictions
        for name in items
    ):
        raise ValueError("Ingredient names must be non-empty strings")
    truth_sets = [set(items) for items in raw_truth]
    prediction_sets = [set(items) for items in raw_predictions]
    if not truth_sets:
        raise ValueError("At least one ground-truth image is required")
    if len(truth_sets) != len(prediction_sets):
        raise ValueError("Ground-truth and prediction image counts must match")
    scenes = list(scene_types) if scene_types is not None else ["unknown"] * len(
        truth_sets
    )
    if len(scenes) != len(truth_sets):
        raise ValueError("Scene types must match the image count")
    if any(not isinstance(scene, str) or not scene.strip() for scene in scenes):
        raise ValueError("Scene types must be non-empty strings")

    classes = (
        list(class_names)
        if class_names is not None
        else sorted(set.union(set(), *truth_sets, *prediction_sets))
    )
    if any(not isinstance(name, str) or not name.strip() for name in classes):
        raise ValueError("Class names must be non-empty strings")
    if len(set(classes)) != len(classes):
        raise ValueError("Class names must be unique non-empty strings")
    observed_classes = set.union(set(), *truth_sets, *prediction_sets)
    unknown_classes = observed_classes - set(classes)
    if unknown_classes:
        raise ValueError(
            "Observed ingredients absent from class_names: "
            + ", ".join(sorted(unknown_classes))
        )

    per_image = [
        _image_scores(truth, prediction)
        for truth, prediction in zip(truth_sets, prediction_sets)
    ]
    scene_scores: dict[str, list[tuple[float, float, float, float]]] = {}
    class_counts = {
        name: [0, 0, 0]
        for name in classes
    }
    for truth, prediction, scene, scores in zip(
        truth_sets, prediction_sets, scenes, per_image
    ):
        scene_scores.setdefault(scene, []).append(scores)
        for name in classes:
            is_true, is_predicted = name in truth, name in prediction
            if is_true and is_predicted:
                class_counts[name][0] += 1
            elif is_predicted:
                class_counts[name][1] += 1
            elif is_true:
                class_counts[name][2] += 1

    per_class: dict[str, ClassMetrics] = {}
    for name, (true_positive, false_positive, false_negative) in class_counts.items():
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
        per_class[name] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            support=true_positive + false_negative,
        )

    return SetMetrics(
        overall=_aggregate(per_image),
        per_class=per_class,
        per_scene_type={
            scene: _aggregate(scores) for scene, scores in sorted(scene_scores.items())
        },
    )
