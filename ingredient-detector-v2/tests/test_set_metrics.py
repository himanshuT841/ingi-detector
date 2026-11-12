import pytest

from src.eval.set_metrics import evaluate_sets


def test_set_metrics_report_image_average_class_and_scene_scores() -> None:
    metrics = evaluate_sets(
        ground_truth=[{"apple", "banana"}, {"apple"}, set()],
        predictions=[{"apple"}, {"apple", "pear"}, {"pear"}],
        scene_types=["counter", "counter", "fridge"],
        class_names=["apple", "banana", "pear", "onion"],
    )

    assert metrics.overall.precision == pytest.approx(0.5)
    assert metrics.overall.recall == pytest.approx(0.5)
    assert metrics.overall.f1 == pytest.approx(4 / 9)
    assert metrics.overall.exact_match == 0
    assert metrics.overall.image_count == 3

    assert metrics.per_class["apple"].f1 == 1
    assert metrics.per_class["apple"].support == 2
    assert metrics.per_class["banana"].false_negative == 1
    assert metrics.per_class["pear"].false_positive == 2
    assert metrics.per_class["onion"].support == 0
    assert metrics.per_class["onion"].f1 == 0

    assert metrics.per_scene_type["counter"].f1 == pytest.approx(2 / 3)
    assert metrics.per_scene_type["counter"].image_count == 2
    assert metrics.per_scene_type["fridge"].f1 == 0


def test_perfect_and_both_empty_images_score_one() -> None:
    metrics = evaluate_sets(
        ground_truth=[{"tomato"}, set()],
        predictions=[{"tomato"}, set()],
        scene_types=["market", "fridge"],
    )

    assert metrics.overall.precision == 1
    assert metrics.overall.recall == 1
    assert metrics.overall.f1 == 1
    assert metrics.overall.exact_match == 1
    assert metrics.per_class["tomato"].support == 1


def test_empty_predictions_for_nonempty_ground_truth_score_zero() -> None:
    metrics = evaluate_sets([{"onion"}], [set()])

    assert metrics.overall.precision == 0
    assert metrics.overall.recall == 0
    assert metrics.overall.f1 == 0
    assert metrics.overall.exact_match == 0


@pytest.mark.parametrize(
    ("truth", "predictions", "scenes", "class_names", "message"),
    [
        ([], [], None, None, "At least one ground-truth"),
        ([{"tomato"}], [], None, None, "image counts must match"),
        ([{"tomato"}], [{"tomato"}], [], None, "Scene types must match"),
        ([{"tomato"}], [{"tomato"}], None, ["onion"], "absent from class_names"),
        ([{"tomato"}], [{"tomato"}], [" "], None, "Scene types must be"),
    ],
)
def test_invalid_metric_inputs_are_rejected(
    truth, predictions, scenes, class_names, message
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_sets(truth, predictions, scenes, class_names)
