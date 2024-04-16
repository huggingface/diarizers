# Adapted from https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/tasks/segmentation/speaker_diarization.py
import math

import numpy as np
import torch

from ..models.segmentation import SegmentationModel


class Preprocess:
    """Converts a HF dataset with the following features:
        - "audio": Audio feature.
        - "speakers": The list of audio speakers, with their order of appearance.
        - "timestamps_start": A list of timestamps indicating the start of each speaker segment.
       flake8>=3.8.3
        - "timestamps_end": A list of timestamps indicating the end of each speaker segment.
    to a preprocessed dataset ready to be used with the HF Trainer.
    """

    def __init__(
        self,
        config,
    ):
        """Preprocess init method.
        Takes as input the dataset to process and the model to perform training with.
        The preprocessing is done to fit the hyperparameters of the model.
        Args:
            input_dataset (dataset): Hugging Face Speaker Diarization dataset
            model (SegmentationModel): A SegmentationModel from the diarizers library.
        """
        self.chunk_duration = config.chunk_duration
        self.max_speakers_per_frame = config.max_speakers_per_frame
        self.max_speakers_per_chunk = config.max_speakers_per_chunk
        self.min_duration = config.min_duration
        self.warm_up = config.warm_up

        self.sample_rate = config.sample_rate
        model = SegmentationModel(config).to_pyannote_model()

        # Get the number of frames associated to a chunk:
        _, self.num_frames_per_chunk, _ = model(torch.rand((1, int(self.chunk_duration * self.sample_rate)))).shape

    def get_labels_in_file(self, file):
        """Get speakers present in file.
        Args:
            file (_type_): dataset row from the input dataset.

        Returns:
            file_labels (list): a list of all speakers in the audio file.
        """

        file_labels = []
        for i in range(len(file["speakers"][0])):
            if file["speakers"][0][i] not in file_labels:
                file_labels.append(file["speakers"][0][i])

        return file_labels

    def get_segments_in_file(self, file, labels):
        """Get segments present in file.

        Args:
            file (_type_): _description_
            labels (_type_): _description_

        Returns:
            annotations (numpy array): _description_
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
        """Method used to get an audio chunk from an audio file given at start_time.

        Args:
            file (dict): dataset row containing the "audio" feature.
            start_time (float): start time (in seconds) of the audio_chunk to extract.

        Returns:
            waveform (array): audio chunk
            y (numpy array): target array.
            labels (list): list of speakers in chunk.
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

        chunk_segments = file_segments[(file_segments["start"] < end_time) & (file_segments["end"] > start_time)]

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

        for start, end, label in zip(start_idx, end_idx, chunk_segments["labels"]):
            mapped_label = mapping[label]
            y[start:end, mapped_label] = 1

        return waveform, y, labels

    def get_start_positions(self, file, overlap, random=False):
        """Get the start positions of the audio_chunks in the input audio file.

        Args:
            file (dict): dataset row containing the "audio" feature.
            overlap (float, optional):  Overlap between successive start positions.
            random (bool, optional):  Whether or not to randomly select chunks in the audio file. Defaults to False.

        Returns:
            start_positions: Numpy array containing the start positions of the audio chunks in file.
        """

        sample_rate = file["audio"][0]["sampling_rate"]

        assert sample_rate == self.sample_rate

        file_duration = len(file["audio"][0]["array"]) / sample_rate
        start_positions = np.arange(0, file_duration - self.chunk_duration, self.chunk_duration * (1 - overlap))

        if random:
            nb_samples = int(file_duration / self.chunk_duration)
            start_positions = np.random.uniform(0, file_duration, nb_samples)

        return start_positions

    def __call__(self, file, random=False, overlap=0.0):
        """Chunk an audio file into short segments of duration self.chunk_duration

        Args:
            file (dict): dataset row containing the "audio" feature.
            random (bool, optional): Whether or not to randomly select chunks in the audio file. Defaults to False.
            overlap (float, optional):  Overlap between successive chunks. Defaults to 0.0.

        Returns:
            new_batch: new batch containing for each chunk the corresponding waveform, labels and number of speakers.
        """

        new_batch = {"waveforms": [], "labels": [], "nb_speakers": []}

        if random:
            start_positions = self.get_start_positions(file, overlap, random=True)
        else:
            start_positions = self.get_start_positions(file, overlap)

        for start_time in start_positions:
            waveform, target, label = self.get_chunk(file, start_time)

            new_batch["waveforms"].append(waveform)
            new_batch["labels"].append(target)
            new_batch["nb_speakers"].append(label)

        return new_batch
