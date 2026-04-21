from pathlib import Path
import json
import numpy as np

from src.data.public_video_loader import load_public_video_dataset


SEQUENCE_LENGTH = 15
RESIZE_TO = (64, 64)
GRAYSCALE = True

# For first tests, keep this small
MAX_VIDEOS = None


def main():
    dataset_root = Path("local_data/public_videos")
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    X_sequences, y_sequences, groups_sequences, label_mapping = load_public_video_dataset(
        dataset_root=dataset_root,
        sequence_length=SEQUENCE_LENGTH,
        resize_to=RESIZE_TO,
        grayscale=GRAYSCALE,
        max_videos=MAX_VIDEOS,
    )

    np.savez_compressed(
        output_dir / "public_sequence_data.npz",
        X_sequences=X_sequences,
        y_sequences=y_sequences,
        groups_sequences=groups_sequences,
    )

    with open(output_dir / "public_label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    print(f"Saved compressed public dataset to: {output_dir / 'public_sequence_data.npz'}")
    print(f"Saved label mapping to: {output_dir / 'public_label_mapping.json'}")


if __name__ == "__main__":
    main()