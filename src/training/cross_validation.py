from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import (
    compute_multilabel_metrics,
    classification_report_df,
    confusion_matrices,
    metrics_dict_to_df,
)


def set_random_seed(random_seed: int = 0):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)


def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def scale_sequences_train_val_test(X_train, X_val, X_test):
    """
    Fit StandardScaler on the training split only and apply it to validation
    and test splits. This avoids leakage from validation or test data.
    """
    scaler = StandardScaler()

    n_train, sequence_length, n_features = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    X_train_scaled = X_train_scaled.reshape(n_train, sequence_length, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, sequence_length, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, sequence_length, n_features)

    return X_train_scaled, X_val_scaled, X_test_scaled


def check_group_overlap(groups_a, groups_b, name_a="split A", name_b="split B"):
    common_groups = np.intersect1d(groups_a, groups_b)
    if len(common_groups) > 0:
        print(f"WARNING: overlap found between {name_a} and {name_b}: {common_groups}")
    else:
        print(f"No overlap between {name_a} and {name_b}.")


def print_label_distribution(y_data, groups_data, fold_type: str):
    print(f"\nLabel distribution in {fold_type}:")
    print(f"  total sequences={len(y_data)}, unique groups={len(np.unique(groups_data))}")

    if y_data.ndim != 2:
        print("  Warning: y_data is expected to be 2D.")
        return

    label_counts = y_data.sum(axis=0)

    for idx, count in enumerate(label_counts):
        print(f"  label_{idx} positives={int(count)}")


def combine_labels_for_stratification(y_sequences: np.ndarray) -> np.ndarray:
    if y_sequences.ndim != 2:
        raise ValueError("y_sequences must be a 2D array")

    n_labels = y_sequences.shape[1]

    if n_labels == 2 and np.array_equal(np.unique(y_sequences), np.array([0, 1])):
        return y_sequences[:, 0] * 2 + y_sequences[:, 1]

    row_sums = y_sequences.sum(axis=1)
    if np.allclose(row_sums, 1.0):
        return np.argmax(y_sequences, axis=1)

    row_strings = ["_".join(map(str, row.astype(int))) for row in y_sequences]
    encoded, _ = pd.factorize(row_strings)
    return encoded


def get_inner_train_val_indices(
    X_train_outer,
    y_train_outer,
    groups_train_outer,
    random_seed: int,
):
    """
    Create one subject-independent train/validation split within the outer
    training fold. The validation split is used only for early stopping.
    """
    combined_y_inner = combine_labels_for_stratification(y_train_outer)

    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_seed)

    inner_train_idx, val_idx = next(
        inner_cv.split(X_train_outer, combined_y_inner, groups_train_outer)
    )

    return inner_train_idx, val_idx


def run_cross_validation(
    X_sequences: np.ndarray,
    y_sequences: np.ndarray,
    groups_sequences: np.ndarray,
    build_model_fn,
    model_params: dict,
    output_dir,
    model_name: str,
    label_names=None,
    n_splits: int = 3,
    epochs: int = 100,
    batch_size: int = 16,
    random_seed: int = 0,
    threshold: float = 0.5,
):
    """
    Run subject-independent StratifiedGroupKFold cross-validation.

    The outer test fold is used only for final evaluation. Early stopping uses
    an inner validation split drawn from the outer training fold. Feature scaling
    is fitted only on the inner training split and then applied to validation and
    outer test data.
    """
    set_random_seed(random_seed)
    output_dir = ensure_output_dir(output_dir)

    if label_names is None:
        label_names = [f"label_{i}" for i in range(y_sequences.shape[1])]

    combined_y = combine_labels_for_stratification(y_sequences)
    outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    all_fold_metrics = []
    y_preds_all_folds = []
    y_tests_all_folds = []
    y_trains_all_folds = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_sequences, combined_y, groups_sequences), start=1
    ):
        print("\n" + "=" * 80)
        print(f"{model_name.upper()} | Fold {fold}/{n_splits}")

        X_train_outer = X_sequences[train_idx]
        X_test = X_sequences[test_idx]
        y_train_outer = y_sequences[train_idx]
        y_test = y_sequences[test_idx]
        groups_train_outer = groups_sequences[train_idx]
        groups_test = groups_sequences[test_idx]

        check_group_overlap(
            groups_train_outer,
            groups_test,
            name_a="outer train",
            name_b="outer test",
        )

        inner_train_idx, val_idx = get_inner_train_val_indices(
            X_train_outer=X_train_outer,
            y_train_outer=y_train_outer,
            groups_train_outer=groups_train_outer,
            random_seed=random_seed + fold,
        )

        X_train = X_train_outer[inner_train_idx]
        X_val = X_train_outer[val_idx]
        y_train = y_train_outer[inner_train_idx]
        y_val = y_train_outer[val_idx]
        groups_train = groups_train_outer[inner_train_idx]
        groups_val = groups_train_outer[val_idx]

        check_group_overlap(
            groups_train,
            groups_val,
            name_a="inner train",
            name_b="inner validation",
        )

        X_train, X_val, X_test = scale_sequences_train_val_test(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
        )

        print_label_distribution(
            y_train,
            groups_train,
            fold_type=f"inner train fold {fold}",
        )
        print_label_distribution(
            y_val,
            groups_val,
            fold_type=f"inner validation fold {fold}",
        )
        print_label_distribution(
            y_test,
            groups_test,
            fold_type=f"outer test fold {fold}",
        )

        y_trains_all_folds.append(y_train)

        model = build_model_fn(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            **model_params,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_binary_accuracy",
                mode="max",
                patience=20,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
        )

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(output_dir / f"{model_name}_history_fold_{fold}.csv", index=False)

        y_pred = model.predict(X_test, verbose=0)
        y_pred_thresholded = (y_pred > threshold).astype(int)

        y_preds_all_folds.append(y_pred)
        y_tests_all_folds.append(y_test)

        fold_metrics = compute_multilabel_metrics(
            y_true=y_test,
            y_pred_thresholded=y_pred_thresholded,
            label_names=label_names,
        )
        all_fold_metrics.append(fold_metrics)

        metrics_df = metrics_dict_to_df(fold_metrics)
        metrics_df.to_csv(
            output_dir / f"{model_name}_metrics_fold_{fold}.csv",
            index=False,
        )

        report_df = classification_report_df(
            y_true=y_test,
            y_pred_thresholded=y_pred_thresholded,
            label_names=label_names,
        )
        report_df.to_csv(
            output_dir / f"{model_name}_classification_report_fold_{fold}.csv",
            index=True,
        )

        cms = confusion_matrices(y_test, y_pred_thresholded)
        for label_idx, label_name in enumerate(label_names):
            cm_df = pd.DataFrame(cms[label_idx])
            cm_df.to_csv(
                output_dir / f"{model_name}_confusion_matrix_{label_name}_fold_{fold}.csv",
                index=False,
            )

        print("\nFold metrics:")
        for measure, value in fold_metrics.items():
            print(f"  {measure}: {value:.4f}")

    aggregated_df = pd.DataFrame(all_fold_metrics)
    aggregated_df.to_csv(output_dir / f"{model_name}_all_fold_metrics.csv", index=False)

    mean_metrics = aggregated_df.mean(axis=0).to_dict()
    mean_metrics_df = metrics_dict_to_df(mean_metrics)
    mean_metrics_df.to_csv(output_dir / f"{model_name}_mean_metrics.csv", index=False)

    np.save(
        output_dir / f"{model_name}_y_preds_all_folds.npy",
        np.array(y_preds_all_folds, dtype=object),
    )
    np.save(
        output_dir / f"{model_name}_y_tests_all_folds.npy",
        np.array(y_tests_all_folds, dtype=object),
    )
    np.save(
        output_dir / f"{model_name}_y_trains_all_folds.npy",
        np.array(y_trains_all_folds, dtype=object),
    )

    print("\n" + "=" * 80)
    print(f"{model_name.upper()} | Mean metrics across folds:")
    for measure, value in mean_metrics.items():
        print(f"  {measure}: {value:.4f}")

    return {
        "all_fold_metrics": aggregated_df,
        "mean_metrics": mean_metrics,
    }