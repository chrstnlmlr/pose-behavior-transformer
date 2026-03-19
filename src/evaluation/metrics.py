import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    multilabel_confusion_matrix,
)


def compute_binary_label_metrics(y_true, y_pred, label_name: str):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    return {
        f"accuracy_{label_name}": accuracy,
        f"precision_{label_name}": precision,
        f"recall_{label_name}": recall,
        f"f1_{label_name}": f1,
        f"specificity_{label_name}": specificity,
    }


def compute_multilabel_metrics(
    y_true,
    y_pred_thresholded,
    label_names=("flap", "jump"),
):
    results = {}

    for i, label_name in enumerate(label_names):
        label_metrics = compute_binary_label_metrics(
            y_true[:, i].ravel(),
            y_pred_thresholded[:, i].ravel(),
            label_name=label_name,
        )
        results.update(label_metrics)

    return results


def classification_report_df(
    y_true,
    y_pred_thresholded,
    label_names=("flap", "jump"),
):
    report = classification_report(
        y_true,
        y_pred_thresholded,
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def confusion_matrices(y_true, y_pred_thresholded):
    return multilabel_confusion_matrix(y_true, y_pred_thresholded)


def metrics_dict_to_df(metrics_dict: dict):
    return pd.DataFrame(list(metrics_dict.items()), columns=["measure", "value"])