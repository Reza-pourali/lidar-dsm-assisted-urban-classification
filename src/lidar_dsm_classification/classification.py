"""Decision Tree / Random Forest classification with and without DSM."""

from dataclasses import dataclass
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier


CLASS_NAMES = ["road", "building", "water", "vegetation", "work field"]
CLASS_LABELS = np.arange(1, 6, dtype=int)


@dataclass
class ModelResult:
    model_name: str
    feature_set: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: np.ndarray
    classification_report: dict
    feature_importances: np.ndarray | None
    prediction: np.ndarray


@dataclass
class ClassificationExperiment:
    without_dsm: dict[str, ModelResult]
    with_dsm: dict[str, ModelResult]


def _validate_inputs(image, roi_labels, train_idx, test_idx):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("Feature image must have shape (rows, cols, bands).")

    rows, cols, _ = image.shape
    roi_labels = np.asarray(roi_labels)
    if roi_labels.ndim != 1 or roi_labels.size != rows * cols:
        raise ValueError("Flattened ROI labels must match image dimensions.")

    for idx in (train_idx, test_idx):
        idx = np.asarray(idx)
        if idx.ndim != 1:
            raise ValueError("train_idx and test_idx must be one-dimensional.")
        if np.any(idx < 0) or np.any(idx >= rows * cols):
            raise ValueError("Sample indices are outside image bounds.")


def _predict_in_chunks(model, features, chunk_size=250_000):
    prediction = np.empty(features.shape[0], dtype=np.int16)
    for start in range(0, features.shape[0], chunk_size):
        stop = min(start + chunk_size, features.shape[0])
        prediction[start:stop] = model.predict(features[start:stop])
    return prediction


def classify_feature_stack(
    image,
    roi_labels,
    train_idx,
    test_idx,
    feature_set,
    random_state=42,
    rf_estimators=100,
    prediction_chunk_size=250_000,
):
    """Train Decision Tree and Random Forest on one feature stack."""
    _validate_inputs(image, roi_labels, train_idx, test_idx)

    rows, cols, bands = image.shape
    features = np.asarray(image, dtype=np.float32).reshape(rows * cols, bands)

    x_train = features[train_idx]
    y_train = roi_labels[train_idx]
    x_test = features[test_idx]
    y_test = roi_labels[test_idx]

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_estimators,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)

        test_pred = model.predict(x_test)
        full_pred = _predict_in_chunks(
            model,
            features,
            chunk_size=prediction_chunk_size,
        ).reshape(rows, cols)

        cm = confusion_matrix(
            y_test,
            test_pred,
            labels=CLASS_LABELS,
        )

        result = ModelResult(
            model_name=name,
            feature_set=feature_set,
            accuracy=float(accuracy_score(y_test, test_pred)),
            macro_f1=float(f1_score(
                y_test,
                test_pred,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )),
            weighted_f1=float(f1_score(
                y_test,
                test_pred,
                labels=CLASS_LABELS,
                average="weighted",
                zero_division=0,
            )),
            confusion_matrix=cm,
            classification_report=classification_report(
                y_test,
                test_pred,
                labels=CLASS_LABELS,
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            ),
            feature_importances=(
                np.asarray(model.feature_importances_, dtype=float)
                if hasattr(model, "feature_importances_")
                else None
            ),
            prediction=full_pred,
        )
        results[name] = result

    return results


def run_experiment(
    image_without_dsm,
    image_with_dsm,
    roi_labels,
    train_idx,
    test_idx,
    random_state=42,
    rf_estimators=100,
):
    """Run identical train/test samples for both feature configurations."""
    image_without_dsm = np.asarray(image_without_dsm)
    image_with_dsm = np.asarray(image_with_dsm)

    if image_without_dsm.shape[:2] != image_with_dsm.shape[:2]:
        raise ValueError("With-DSM and without-DSM rasters must share rows and columns.")
    if image_with_dsm.shape[2] != image_without_dsm.shape[2] + 1:
        raise ValueError(
            "Expected the with-DSM stack to contain exactly one additional band."
        )

    without = classify_feature_stack(
        image_without_dsm,
        roi_labels,
        train_idx,
        test_idx,
        feature_set="without DSM",
        random_state=random_state,
        rf_estimators=rf_estimators,
    )

    with_dsm = classify_feature_stack(
        image_with_dsm,
        roi_labels,
        train_idx,
        test_idx,
        feature_set="with DSM",
        random_state=random_state,
        rf_estimators=rf_estimators,
    )

    return ClassificationExperiment(
        without_dsm=without,
        with_dsm=with_dsm,
    )
