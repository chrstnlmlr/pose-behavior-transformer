from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def load_fold_arrays(model_name: str):
    output_dir = Path("outputs") / model_name

    y_preds = np.load(output_dir / f"{model_name}_y_preds_all_folds.npy", allow_pickle=True)
    y_tests = np.load(output_dir / f"{model_name}_y_tests_all_folds.npy", allow_pickle=True)

    y_pred_all = np.concatenate(list(y_preds), axis=0)
    y_test_all = np.concatenate(list(y_tests), axis=0)

    return y_pred_all, y_test_all


def compute_threshold_metrics(y_pred, y_true, label_index: int, label_name: str, thresholds):
    rows = []

    y_true_label = y_true[:, label_index]

    for threshold in thresholds:
        y_pred_label = (y_pred[:, label_index] > threshold).astype(int)

        rows.append(
            {
                "label": label_name,
                "threshold": threshold,
                "precision": precision_score(y_true_label, y_pred_label, zero_division=0),
                "recall": recall_score(y_true_label, y_pred_label, zero_division=0),
                "f1": f1_score(y_true_label, y_pred_label, zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def plot_threshold_curve(df, model_name: str, label_name: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    ax.plot(df["threshold"], df["recall"], marker="o", label="Recall")
    ax.plot(df["threshold"], df["f1"], marker="o", label="F1")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name.upper()} threshold analysis: {label_name}")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    thresholds = np.arange(0.1, 1.0, 0.1)

    # label index 0 = flap, 1 = jump
    label_index = 1
    label_name = "jump"

    all_results = []

    for model_name in ["lstm", "transformer"]:
        y_pred, y_true = load_fold_arrays(model_name)

        df = compute_threshold_metrics(
            y_pred=y_pred,
            y_true=y_true,
            label_index=label_index,
            label_name=label_name,
            thresholds=thresholds,
        )
        df["model"] = model_name

        output_csv = reports_dir / f"threshold_analysis_{label_name}_{model_name}.csv"
        output_png = reports_dir / f"figure_threshold_{label_name}_{model_name}.png"

        df.to_csv(output_csv, index=False)
        plot_threshold_curve(df, model_name, label_name, output_png)

        all_results.append(df)

        print(f"\n{model_name.upper()} threshold analysis:")
        print(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(reports_dir / f"threshold_analysis_{label_name}_combined.csv", index=False)

    print("\nSaved threshold analysis files to reports/")


if __name__ == "__main__":
    main()