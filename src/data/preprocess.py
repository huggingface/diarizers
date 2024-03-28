import argparse
import math

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from pyannote.audio.models.segmentation import PyanNet
import torch
from pyannote.audio.core.task import Problem, Resolution, Specifications


class Preprocess: 

    def __init__(
        self, 
        duration=10, 
        max_speakers_per_frame=2, 
        max_speakers_per_chunk=3, 
        min_duration=None, 
        warm_up=(0.0, 0.0), 
    ): 
        
        self.model = PyanNet(sincnet={"stride": 10})
    
        self.model.specifications = Specifications(
                    problem=Problem.MULTI_LABEL_CLASSIFICATION
                    if max_speakers_per_frame is None
                    else Problem.MONO_LABEL_CLASSIFICATION,
                    resolution=Resolution.FRAME,
                    duration=duration,
                    min_duration=min_duration,
                    warm_up=warm_up,
                    classes=[f"speaker#{i+1}" for i in range(max_speakers_per_chunk)],
                    powerset_max_classes=max_speakers_per_frame,
                    permutation_invariant=True,
                )
        self.model.build()

        # Get the number of frames associated to a chunk:
        sample_rate = 16000
        _, self.num_frames_per_chunk, _ = self.model(torch.rand((1, int(duration * sample_rate)))).shape


    def get_labels_in_file(self, file):
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


    def get_segments_in_file(self, file, labels):
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


    def get_chunk(self, file, start_time, duration):
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
        num_frames_waveform = math.floor(duration * sample_rate)
        end_frame = start_frame + num_frames_waveform

        waveform = file["audio"][0]["array"][start_frame:end_frame]

        labels = self.get_labels_in_file(file)

        file_segments = self.get_segments_in_file(file, labels)

        chunk_segments = file_segments[
            (file_segments["start"] < end_time) & (file_segments["end"] > start_time)
        ]

        # compute frame resolution:
        resolution = duration / self.num_frames_per_chunk

        # discretize chunk annotations at model output resolution
        start = np.maximum(chunk_segments["start"], start_time) - start_time
        start_idx = np.floor(start / resolution).astype(int)
        end = np.minimum(chunk_segments["end"], end_time) - start_time
        end_idx = np.ceil(end / resolution).astype(int)

        # get list and number of labels for current scope
        labels = list(np.unique(chunk_segments["labels"]))
        num_labels = len(labels)
        # initial frame-level targets
        y = np.zeros((self.num_frames_per_chunk, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}

        for start, end, label in zip(
            start_idx, end_idx, chunk_segments["labels"]
        ):
            mapped_label = mapping[label]
            y[start:end, mapped_label] = 1

        return waveform, y, labels


    def get_start_positions(self, file, duration, overlap, random=False):
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


    def chunk_file(self, file, duration=2, select_random=False, overlap=0.0):
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
            start_positions = self.get_start_positions(file, duration, overlap, random=True)
        else:
            start_positions = self.get_start_positions(file, duration, overlap)

        for start_time in start_positions:

            waveform, target, label = self.get_chunk(file, start_time, duration)

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

    preprocess = Preprocess(chunk_duration)

    processed_spd_dataset["train"] = ds["train"].map(
        lambda file: preprocess.chunk_file(
            file, duration=chunk_duration, select_random=False, overlap=0.5
        ),
        batched=True,
        batch_size=1,
        remove_columns=ds["train"].column_names,
        num_proc=1,
    )
    processed_spd_dataset["train"] = processed_spd_dataset["train"].shuffle(seed=42)

    processed_spd_dataset["validation"] = ds["validation"].map(
        lambda file: preprocess.chunk_file(
            file, duration=chunk_duration, select_random=False, overlap=0.0
        ),
        batched=True,
        batch_size=1,
        remove_columns=ds["validation"].column_names,
        num_proc=24,
    )

    processed_spd_dataset["test"] = ds["test"].map(
        lambda file: preprocess.chunk_file(
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

    ds = load_dataset("kamilakesbi/ami_spd_nobatch", num_proc=12)

    processed_dataset = preprocess_spd_dataset(
        ds, chunk_duration=int(args.chunk_duration)
    )

    processed_dataset.push_to_hub("kamilakesbi/real_ami_processed_sc2")
