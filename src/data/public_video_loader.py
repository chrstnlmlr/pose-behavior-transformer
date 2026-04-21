from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def find_video_files(dataset_root: Path) -> List[Path]:
    video_files = []
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(path)
    return sorted(video_files)


def infer_label_from_parent_folder(video_path: Path) -> str:
    """
    Assumes folder structure like:
    dataset_root/<label_name>/<video_file>
    """
    return video_path.parent.name


def sample_frame_indices(num_frames: int, sequence_length: int) -> np.ndarray:
    if num_frames < sequence_length:
        # repeat last frame if too short
        idx = np.linspace(0, max(num_frames - 1, 0), num=sequence_length)
    else:
        idx = np.linspace(0, num_frames - 1, num=sequence_length)
    return np.round(idx).astype(int)


def load_video_frames(
    video_path: Path,
    sequence_length: int = 15,
    resize_to: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
) -> np.ndarray:
    """
    Returns array of shape:
    - grayscale: (sequence_length, H*W)
    - color: (sequence_length, H*W*3)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    frame_indices = sample_frame_indices(len(frames), sequence_length)

    processed_frames = []
    for idx in frame_indices:
        frame = frames[idx]

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = frame.flatten()

        processed_frames.append(frame)

    return np.stack(processed_frames, axis=0)


def build_label_mapping(video_files: List[Path]) -> dict:
    labels = sorted({infer_label_from_parent_folder(v) for v in video_files})
    return {label: idx for idx, label in enumerate(labels)}


def one_hot_encode(label_idx: int, num_classes: int) -> np.ndarray:
    y = np.zeros(num_classes, dtype=np.float32)
    y[label_idx] = 1.0
    return y


def load_public_video_dataset(
    dataset_root: Path,
    sequence_length: int = 15,
    resize_to: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
    max_videos: int = None,
):
    """
    Loads videos from folder-structured dataset into sequence arrays.

    Expected structure:
    dataset_root/
        class_a/
            video1.avi
            video2.avi
        class_b/
            video3.avi
            ...

    Returns
    -------
    X_sequences : np.ndarray
        Shape (N, sequence_length, feature_dim)
    y_sequences : np.ndarray
        Shape (N, num_classes)
    groups_sequences : np.ndarray
        Shape (N,), here simply video ids / paths
    label_mapping : dict
        class name -> integer index
    """
    dataset_root = Path(dataset_root)
    video_files = find_video_files(dataset_root)

    if len(video_files) == 0:
        raise FileNotFoundError(f"No videos found in: {dataset_root}")

    if max_videos is not None:
        video_files = video_files[:max_videos]

    label_mapping = build_label_mapping(video_files)
    num_classes = len(label_mapping)

    X_list = []
    y_list = []
    groups_list = []

    for video_path in video_files:
        label_name = infer_label_from_parent_folder(video_path)
        label_idx = label_mapping[label_name]

        try:
            x_seq = load_video_frames(
                video_path=video_path,
                sequence_length=sequence_length,
                resize_to=resize_to,
                grayscale=grayscale,
            )
        except Exception as e:
            print(f"Skipping {video_path.name}: {e}")
            continue

        y_seq = one_hot_encode(label_idx, num_classes=num_classes)
        group_id = video_path.stem

        X_list.append(x_seq)
        y_list.append(y_seq)
        groups_list.append(group_id)

    X_sequences = np.stack(X_list, axis=0).astype(np.float32)
    y_sequences = np.stack(y_list, axis=0).astype(np.float32)
    groups_sequences = np.array(groups_list, dtype=object)

    print(f"Loaded public dataset from: {dataset_root}")
    print(f"X_sequences shape: {X_sequences.shape}")
    print(f"y_sequences shape: {y_sequences.shape}")
    print(f"groups_sequences shape: {groups_sequences.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_mapping}")

    return X_sequences, y_sequences, groups_sequences, label_mapping