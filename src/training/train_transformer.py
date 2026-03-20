from pathlib import Path
import numpy as np

from src.models.transformer_model import build_transformer_model
from src.training.cross_validation import run_cross_validation


def load_sequence_data(npz_path: str | Path):
    data = np.load(npz_path, allow_pickle=True)
    return data["X_sequences"], data["y_sequences"], data["groups_sequences"]


def main():
    data_path = Path("data/sequence_data.npz")
    output_dir = Path("outputs/transformer")

    X_sequences, y_sequences, groups_sequences = load_sequence_data(data_path)

    model_params = {
        "d_model": 128,
        "num_heads": 4,
        "ff_dim": 256,
        "num_layers": 2,
        "dropout_rate": 0.1,
        "learning_rate": 0.0003,
        "output_units": 2,
    }

    run_cross_validation(
        X_sequences=X_sequences,
        y_sequences=y_sequences,
        groups_sequences=groups_sequences,
        build_model_fn=build_transformer_model,
        model_params=model_params,
        output_dir=output_dir,
        model_name="transformer",
        label_names=("flap", "jump"),
        n_splits=3,
        epochs=100,
        batch_size=16,
        random_seed=0,
    )


if __name__ == "__main__":
    main()