from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_history_files(model_name: str, base_dir: str | Path):
    base_dir = Path(base_dir)
    pattern = f"{model_name}_history_fold_*.csv"
    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No history files found for pattern: {base_dir / pattern}")
    return files


def plot_metric_across_folds(history_files, metric_name: str, output_path: str | Path, title: str):
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for history_file in history_files:
        df = pd.read_csv(history_file)
        fold_name = history_file.stem.replace("_history_", " ").replace("_", " ")

        if metric_name not in df.columns:
            print(f"Skipping {history_file.name}: metric '{metric_name}' not found")
            continue

        ax.plot(df[metric_name], label=fold_name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    output_root = Path("outputs")

    # LSTM
    lstm_dir = output_root / "lstm"
    lstm_history_files = find_history_files("lstm", lstm_dir)

    plot_metric_across_folds(
        history_files=lstm_history_files,
        metric_name="loss",
        output_path=lstm_dir / "lstm_loss_curves.png",
        title="LSTM Training Loss Across Folds",
    )
    plot_metric_across_folds(
        history_files=lstm_history_files,
        metric_name="val_loss",
        output_path=lstm_dir / "lstm_val_loss_curves.png",
        title="LSTM Validation Loss Across Folds",
    )
    plot_metric_across_folds(
        history_files=lstm_history_files,
        metric_name="binary_accuracy",
        output_path=lstm_dir / "lstm_accuracy_curves.png",
        title="LSTM Training Accuracy Across Folds",
    )
    plot_metric_across_folds(
        history_files=lstm_history_files,
        metric_name="val_binary_accuracy",
        output_path=lstm_dir / "lstm_val_accuracy_curves.png",
        title="LSTM Validation Accuracy Across Folds",
    )

    # Transformer
    transformer_dir = output_root / "transformer"
    transformer_history_files = find_history_files("transformer", transformer_dir)

    plot_metric_across_folds(
        history_files=transformer_history_files,
        metric_name="loss",
        output_path=transformer_dir / "transformer_loss_curves.png",
        title="Transformer Training Loss Across Folds",
    )
    plot_metric_across_folds(
        history_files=transformer_history_files,
        metric_name="val_loss",
        output_path=transformer_dir / "transformer_val_loss_curves.png",
        title="Transformer Validation Loss Across Folds",
    )
    plot_metric_across_folds(
        history_files=transformer_history_files,
        metric_name="binary_accuracy",
        output_path=transformer_dir / "transformer_accuracy_curves.png",
        title="Transformer Training Accuracy Across Folds",
    )
    plot_metric_across_folds(
        history_files=transformer_history_files,
        metric_name="val_binary_accuracy",
        output_path=transformer_dir / "transformer_val_accuracy_curves.png",
        title="Transformer Validation Accuracy Across Folds",
    )

    print("Saved training curve plots for LSTM and Transformer.")


if __name__ == "__main__":
    main()