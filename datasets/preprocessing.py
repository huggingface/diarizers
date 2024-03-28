import argparse
import math

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from pyannote.audio.models.segmentation import PyanNet


def get_labels_in_file(file):
    """Get speakers
    Args:
        file (_type_): _description_

    Returns:
        _type_: _description_
    """

    file_labels = []
    for i in range(len(file["speakers"][0])):

        if file["speakers"][0][i] not in file_labels:
            file_labels.append(file["speakers"][0][i])

    return file_labels


def get_segments_in_file(file, labels):
    """_summary_

    Args:
        file (_type_): _description_
        labels (_type_): _description_

    Returns:
        _type_: _description_
    """

    file_annotations = []

    for i in range(len(file["timestamps_start"][0])):

        start_segment = file["timestamps_start"][0][i]
        end_segment = file["timestamps_end"][0][i]
        label = labels.index(file["speakers"][0][i])
        file_annotations.append((start_segment, end_segment, label))

    dtype = [("start", "<f4"), ("end", "<f4"), ("labels", "i1")]

    annotations = np.array(file_annotations, dtype)

    return annotations


def get_chunk(file, start_time, duration):
    """_summary_

    Args:
        file (_type_): _description_
        start_time (_type_): _description_
        duration (_type_): _description_

    Returns:
        _type_: _description_
    """

    sample_rate = file["audio"][0]["sampling_rate"]
    end_time = start_time + duration
    start_frame = math.floor(start_time * sample_rate)
    num_frames = math.floor(duration * sample_rate)
    end_frame = start_frame + num_frames

    waveform = file["audio"][0]["array"][start_frame:end_frame]

    labels = get_labels_in_file(file)

    file_segments = get_segments_in_file(file, labels)

    chunk_segments = file_segments[
        (file_segments["start"] < end_time) & (file_segments["end"] > start_time)
    ]

    model = PyanNet(sincnet={"stride": 10})
    step = model.receptive_field.step
    half = 0.5 * model.receptive_field.duration

    start = np.maximum(chunk_segments["start"], start_time) - start_time - half
    start_idx = np.maximum(0, np.round(start / step)).astype(int)

    end = np.minimum(chunk_segments["end"], end_time) - start_time - half
    end_idx = np.round(end / step).astype(int)

    labels = list(np.unique(chunk_segments["labels"]))
    num_labels = len(labels)

    num_frames = model.num_frames(round(duration * model.hparams.sample_rate))
    y = np.zeros((num_frames, num_labels), dtype=np.uint8)

    mapping = {label: idx for idx, label in enumerate(labels)}

    for start, end, label in zip(start_idx, end_idx, chunk_segments["labels"]):

        mapped_label = mapping[label]
        y[start : end + 1, mapped_label] = 1

    return waveform, y, labels


def get_start_positions(file, duration, overlap, random=False):
    """_summary_

    Args:
        file (_type_): _description_
        duration (_type_): _description_
        overlap (_type_): _description_

    Returns:
        _type_: _description_
    """

    sample_rate = file["audio"][0]["sampling_rate"]
    file_duration = len(file["audio"][0]["array"]) / sample_rate
    start_positions = np.arange(0, file_duration - duration, duration * (1 - overlap))

    if random:

        nb_samples = int(file_duration / duration)
        start_positions = np.random.uniform(0, file_duration, nb_samples)

    return start_positions


def chunk_file(file, duration=2, select_random=False, overlap=0.0):
    """_summary_

    Args:
        file (_type_): _description_
        duration (int, optional): _description_. Defaults to 2.
        select_random (bool, optional): _description_. Defaults to False.
        overlap (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """

    new_batch = {"waveforms": [], "labels": [], "nb_speakers": []}

    if select_random:
        start_positions = get_start_positions(file, duration, overlap, random=True)
    else:
        start_positions = get_start_positions(file, duration, overlap)

    for start_time in start_positions:

        waveform, target, label = get_chunk(file, start_time, duration)

        new_batch["waveforms"].append(waveform)
        new_batch["labels"].append(target)
        new_batch["nb_speakers"].append(label)

    return new_batch


def preprocess_spd_dataset(ds, chunk_duration):
    """_summary_

    Args:
        ds (_type_): _description_
        chunk_duration (_type_): _description_

    Returns:
        _type_: _description_
    """

    processed_spd_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({}),
            "validation": Dataset.from_dict({}),
            "test": Dataset.from_dict({}),
        }
    )

    processed_spd_dataset["train"] = ds["train"].map(
        lambda file: chunk_file(
            file, duration=chunk_duration, select_random=False, overlap=0.5
        ),
        batched=True,
        batch_size=1,
        remove_columns=ds["train"].column_names,
        num_proc=24,
    )
    processed_spd_dataset["train"] = processed_spd_dataset["train"].shuffle(seed=42)

    processed_spd_dataset["validation"] = ds["validation"].map(
        lambda file: chunk_file(
            file, duration=chunk_duration, select_random=False, overlap=0.0
        ),
        batched=True,
        batch_size=1,
        remove_columns=ds["validation"].column_names,
        num_proc=24,
    )

    processed_spd_dataset["test"] = ds["test"].map(
        lambda file: chunk_file(
            file, duration=chunk_duration, select_random=False, overlap=0.75
        ),
        batched=True,
        batch_size=1,
        remove_columns=ds["validation"].column_names,
        num_proc=24,
    )

    return processed_spd_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_duration", help="", default="10")

    args = parser.parse_args()

    ds = load_dataset("kamilakesbi/ami_spd_augmented_silences2", num_proc=12)

    processed_dataset = preprocess_spd_dataset(
        ds, chunk_duration=int(args.chunk_duration)
    )

    processed_dataset.push_to_hub("kamilakesbi/ami_spd_augmented_silences_processed2")
