from pathlib import Path
import numpy as np

from src.models.transformer_model import build_transformer_model
from src.training.cross_validation import run_cross_validation


def load_sequence_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["X_sequences"], data["y_sequences"], data["groups_sequences"]


def main():
    data_path = Path("data/public_sequence_data.npz")
    output_dir = Path("outputs/public_transformer")

    X_sequences, y_sequences, groups_sequences = load_sequence_data(data_path)

    model_params = {
        "d_model": 128,
        "num_heads": 4,
        "ff_dim": 256,
        "num_layers": 2,
        "dropout_rate": 0.1,
        "learning_rate": 0.0003,
        "output_units": y_sequences.shape[1],
    }

    label_names = [f"class_{i}" for i in range(y_sequences.shape[1])]

    run_cross_validation(
        X_sequences=X_sequences,
        y_sequences=y_sequences,
        groups_sequences=groups_sequences,
        build_model_fn=build_transformer_model,
        model_params=model_params,
        output_dir=output_dir,
        model_name="public_transformer",
        label_names=label_names,
        n_splits=3,
        epochs=50,
        batch_size=16,
        random_seed=0,
    )


if __name__ == "__main__":
    main()