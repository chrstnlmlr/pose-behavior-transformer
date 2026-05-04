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


def plot_cm(cm, save_path):
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

    # cell borders
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    cm_array = load_cm("transformer", "jump")
    cm_mean = np.mean(cm_array, axis=0)

    plot_cm(
        cm_mean,
        reports_dir / "figure_cm_jump_transformer.png"
    )

    print("Saved confusion matrix to reports/")


if __name__ == "__main__":
    main()