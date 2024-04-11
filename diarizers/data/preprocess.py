import math

import numpy as np
from datasets import Dataset, DatasetDict
import torch

class Preprocess: 

    def __init__(
        self, 
        input_dataset, 
        model, 
    ): 
        
        self.input_dataset = input_dataset
        self.chunk_duration = model.chunk_duration
        self.max_speakers_per_frame = model.max_speakers_per_frame
        self.max_speakers_per_chunk = model.max_speakers_per_chunk
        self.min_duration = model.min_duration
        self.warm_up = model.warm_up

        self.model = model
        self.model = self.model.to_pyannote_model()

        # Get the number of frames associated to a chunk:
        self.sample_rate = input_dataset['train'][0]['audio']['sampling_rate']

        _, self.num_frames_per_chunk, _ = self.model(torch.rand((1, int(self.chunk_duration * self.sample_rate)))).shape


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


    def get_chunk(self, file, start_time):
        """_summary_

        Args:
            file (_type_): _description_
            start_time (_type_): _description_
            duration (_type_): _description_

        Returns:
            _type_: _description_
        """

        sample_rate = file["audio"][0]["sampling_rate"]

        assert sample_rate == self.sample_rate

        end_time = start_time + self.chunk_duration
        start_frame = math.floor(start_time * sample_rate)
        num_frames_waveform = math.floor(self.chunk_duration * sample_rate)
        end_frame = start_frame + num_frames_waveform

        waveform = file["audio"][0]["array"][start_frame:end_frame]

        labels = self.get_labels_in_file(file)

        file_segments = self.get_segments_in_file(file, labels)

        chunk_segments = file_segments[
            (file_segments["start"] < end_time) & (file_segments["end"] > start_time)
        ]

        # compute frame resolution:
        resolution = self.chunk_duration / self.num_frames_per_chunk

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


    def get_start_positions(self, file, overlap, random=False):
        """_summary_

        Args:
            file (_type_): _description_
            duration (_type_): _description_
            overlap (_type_): _description_

        Returns:
            _type_: _description_
        """

        sample_rate = file["audio"][0]["sampling_rate"]

        assert sample_rate == self.sample_rate

        file_duration = len(file["audio"][0]["array"]) / sample_rate
        start_positions = np.arange(0, file_duration - self.chunk_duration, self.chunk_duration * (1 - overlap))

        if random:
            nb_samples = int(file_duration / self.chunk_duration)
            start_positions = np.random.uniform(0, file_duration, nb_samples)

        return start_positions


    def chunk_file(self, file, select_random=False, overlap=0.0):
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
            start_positions = self.get_start_positions(file, overlap, random=True)
        else:
            start_positions = self.get_start_positions(file, overlap)

        for start_time in start_positions:

            waveform, target, label = self.get_chunk(file, start_time)

            new_batch["waveforms"].append(waveform)
            new_batch["labels"].append(target)
            new_batch["nb_speakers"].append(label)

        return new_batch


    def preprocess_dataset(self, num_proc=1):
        """_summary_

        Args:
            ds (_type_): _description_
            chunk_duration (_type_): _description_

        Returns:
            _type_: _description_
        """

        self.processed_dataset = DatasetDict(
            {
                "train": Dataset.from_dict({}),
                "validation": Dataset.from_dict({}),
                "test": Dataset.from_dict({}),
            }
        )

        self.processed_dataset["train"] = self.input_dataset["train"].map(
            lambda file: self.chunk_file(
                file, select_random=False, overlap=0.5
            ),
            batched=True,
            batch_size=1,
            remove_columns=self.input_dataset["train"].column_names,
            num_proc=num_proc,
        )
        self.processed_dataset["train"] = self.processed_dataset["train"].shuffle(seed=42)

        self.processed_dataset["validation"] = self.input_dataset["validation"].map(
            lambda file: self.chunk_file(
                file, select_random=False, overlap=0.0
            ),
            batched=True,
            batch_size=1,
            remove_columns=self.input_dataset["validation"].column_names,
            num_proc=num_proc,
        )

        self.processed_dataset["test"] = self.input_dataset["test"].map(
            lambda file: self.chunk_file(
                file, select_random=False, overlap=0.75
            ),
            batched=True,
            batch_size=1,
            remove_columns=self.input_dataset["validation"].column_names,
            num_proc=num_proc,
        )

        return self.processed_dataset
