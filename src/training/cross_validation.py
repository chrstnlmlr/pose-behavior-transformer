from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

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


def ensure_output_dir(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def check_group_overlap(groups_train, groups_test):
    common_groups = np.intersect1d(groups_train, groups_test)
    if len(common_groups) > 0:
        print(f"WARNING: overlap found in groups: {common_groups}")
    else:
        print("No overlap in groups between training and test sets.")


def print_label_distribution(y_data, groups_data, fold_type: str):
    no_man = np.sum(np.all(y_data == [0, 0], axis=1))
    jump = np.sum(np.all(y_data == [0, 1], axis=1))
    flap = np.sum(np.all(y_data == [1, 0], axis=1))
    flap_jump = np.sum(np.all(y_data == [1, 1], axis=1))
    group_count = len(np.unique(groups_data))

    print(f"\nLabel distribution in {fold_type}:")
    print(
        f"  no_man={no_man}, jump={jump}, flap={flap}, flap_jump={flap_jump}"
    )
    print(f"  total sequences={len(y_data)}, unique groups={group_count}")


def combine_multilabel_for_stratification(y_sequences: np.ndarray):
    return y_sequences[:, 0] * 2 + y_sequences[:, 1]


def run_cross_validation(
    X_sequences: np.ndarray,
    y_sequences: np.ndarray,
    groups_sequences: np.ndarray,
    build_model_fn,
    model_params: dict,
    output_dir: str | Path,
    model_name: str,
    label_names=("flap", "jump"),
    n_splits: int = 3,
    epochs: int = 100,
    batch_size: int = 16,
    random_seed: int = 0,
    threshold: float = 0.5,
):
    """
    Runs subject-independent StratifiedGroupKFold CV for a model builder function.

    Parameters
    ----------
    X_sequences : np.ndarray
        Shape: (n_sequences, sequence_length, n_features)
    y_sequences : np.ndarray
        Shape: (n_sequences, n_labels)
    groups_sequences : np.ndarray
        Shape: (n_sequences,)
    build_model_fn : callable
        Function returning a compiled model. Must accept `input_shape`.
    model_params : dict
        Parameters forwarded to build_model_fn besides input_shape.
    output_dir : str | Path
        Directory for fold results.
    model_name : str
        Used for folder naming and logs.
    """
    set_random_seed(random_seed)
    output_dir = ensure_output_dir(output_dir)

    combined_y = combine_multilabel_for_stratification(y_sequences)
    cv = StratifiedGroupKFold(n_splits=n_splits)

    all_fold_metrics = []
    y_preds_all_folds = []
    y_tests_all_folds = []
    y_trains_all_folds = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X_sequences, combined_y, groups_sequences), start=1
    ):
        print("\n" + "=" * 80)
        print(f"{model_name.upper()} | Fold {fold}/{n_splits}")

        X_train, X_test = X_sequences[train_idx], X_sequences[test_idx]
        y_train, y_test = y_sequences[train_idx], y_sequences[test_idx]
        groups_train, groups_test = groups_sequences[train_idx], groups_sequences[test_idx]

        check_group_overlap(groups_train, groups_test)
        print_label_distribution(y_train, groups_train, fold_type=f"train fold {fold}")
        print_label_distribution(y_test, groups_test, fold_type=f"test fold {fold}")

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
            validation_data=(X_test, y_test),
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

    np.save(output_dir / f"{model_name}_y_preds_all_folds.npy", np.array(y_preds_all_folds, dtype=object))
    np.save(output_dir / f"{model_name}_y_tests_all_folds.npy", np.array(y_tests_all_folds, dtype=object))
    np.save(output_dir / f"{model_name}_y_trains_all_folds.npy", np.array(y_trains_all_folds, dtype=object))

    print("\n" + "=" * 80)
    print(f"{model_name.upper()} | Mean metrics across folds:")
    for measure, value in mean_metrics.items():
        print(f"  {measure}: {value:.4f}")

    return {
        "all_fold_metrics": aggregated_df,
        "mean_metrics": mean_metrics,
    }