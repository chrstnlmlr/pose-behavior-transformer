from pathlib import Path
import numpy as np

from src.models.lstm_model import build_lstm_model
from src.training.cross_validation import run_cross_validation


def load_sequence_data(npz_path: str | Path):
    data = np.load(npz_path, allow_pickle=True)
    return data["X_sequences"], data["y_sequences"], data["groups_sequences"]


def main():
    data_path = Path("data/sequence_data.npz")
    output_dir = Path("outputs/lstm")

    X_sequences, y_sequences, groups_sequences = load_sequence_data(data_path)

    model_params = {
        "num_layers": 3,
        "units_layer_1": 256,
        "units_layer_2": 256,
        "units_layer_3": 256,
        "dropout_rate": 0.1,
        "learning_rate": 0.0003,
        "output_units": 2,
    }

    run_cross_validation(
        X_sequences=X_sequences,
        y_sequences=y_sequences,
        groups_sequences=groups_sequences,
        build_model_fn=build_lstm_model,
        model_params=model_params,
        output_dir=output_dir,
        model_name="lstm",
        label_names=("flap", "jump"),
        n_splits=3,
        epochs=100,
        batch_size=16,
        random_seed=0,
    )


if __name__ == "__main__":
    main()