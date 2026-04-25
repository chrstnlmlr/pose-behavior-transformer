from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_cm(model_name, label):
    pattern = f"{model_name}_confusion_matrix_{label}_fold_*.csv"
    files = sorted(Path("outputs").rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    matrices = []
    for f in files:
        df = pd.read_csv(f)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        matrices.append(df.values.astype(float))

    return np.array(matrices)


def compute_mean_cm(cm_array):
    return np.mean(cm_array, axis=0)


def plot_cm(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    for label in ["flap", "jump"]:
        for model in ["lstm", "transformer"]:
            cm_array = load_cm(model, label)
            cm_mean = compute_mean_cm(cm_array)

            plot_cm(
                cm_mean,
                f"{model.upper()} - {label}",
                reports_dir / f"figure_cm_{label}_{model}.png"
            )

    print("Saved all confusion matrices to reports/")


if __name__ == "__main__":
    main()