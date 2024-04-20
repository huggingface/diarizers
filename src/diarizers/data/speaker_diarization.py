# Adapted from https://github.com/hbredin/pyannote-db-callhome/blob/master/parse_transcripts.py
# The MIT License (MIT)

# Copyright (c) 2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from datasets import Audio, Dataset, DatasetDict


def get_secs(x):
    return x * 4 * 2.0 / 8000


def get_start_end(t1, t2):
    t1 = get_secs(t1)
    t2 = get_secs(t2)
    return t1, t2


def represent_int(s):
    try:
        int(s)
        return True
    except ValueError as e:
        return False


class SpeakerDiarizationDataset:
    """
    Convert a speaker diarization dataset made of <audio files, annotations files>
    into a HF dataset with the following features:
        - "audio": Audio feature.
        - "speakers": The list of audio speakers, with their order of appearance.
        - "timestamps_start": A list of timestamps indicating the start of each speaker segment.
        - "timestamps_end": A list of timestamps indicating the end of each speaker segment.
    """

    def __init__(
        self,
        audio_paths,
        annotations_paths,
        sample_rate=16000,
        annotations_type="rttm",
        crop_unannotated_regions=False,
    ):
        """
        Args:
            audio_paths (dict): A dict with keys (str): split subset - example: "train" and values: list of str paths to audio files.
            annotations_paths (dict): A dict with keys (str): split subset - example: "train" and values: list of str paths to annotations files.
            sample_rate (int, optional): Audios sampling rate in the generated HF dataset. Defaults to 16000.
        """
        annotations_type in ["rttm", "cha"]
        self.audio_paths = audio_paths
        self.annotations_paths = annotations_paths
        self.sample_rate = sample_rate
        self.annotations_type = annotations_type
        self.crop_unannotated_regions = crop_unannotated_regions

    def crop_audio(self, files):
        # Load audio from path
        new_batch = {
            "audio": [],
            "timestamps_start": [],
            "timestamps_end": [],
            "speakers": [],
        }

        batch = [{key: values[i] for key, values in files.items()} for i in range(len(files["audio"]))]

        for file in batch:
            # Crop audio based on timestamps (in samples)

            # We add a file only if it's annotated:
            if len(file["timestamps_start"]) != 0:
                start_idx = int(file["timestamps_start"][0] * self.sample_rate)
                end_idx = int(max(file["timestamps_end"]) * self.sample_rate)

                waveform = file["audio"]["array"]

                audio = {
                    "array": np.array(waveform[start_idx:end_idx]),
                    "sampling_rate": self.sample_rate,
                }

                timestamps_start = [start - file["timestamps_start"][0] for start in file["timestamps_start"]]
                timestamps_end = [end - file["timestamps_start"][0] for end in file["timestamps_end"]]

                new_batch["audio"].append(audio)
                new_batch["timestamps_start"].append(timestamps_start)
                new_batch["timestamps_end"].append(timestamps_end)
                new_batch["speakers"].append(file["speakers"])

        return new_batch

    def process_cha_file(self, path_to_cha):
        timestamps_start = []
        timestamps_end = []
        speakers = []

        line = open(path_to_cha, "r").read().splitlines()
        for i, line in enumerate(line):
            if line.startswith("*"):
                id = line.split(":")[0][1:]
            splits = line.split(" ")
            if splits[-1].find("_") != -1:
                indexes = splits[-1].strip()
                start = indexes.split("_")[0].strip()[1:]
                end = indexes.split("_")[1].strip()[:-1]
                if represent_int(start) and represent_int(end):
                    start, end = get_start_end(int(start), int(end))

                    speakers.append(id)
                    timestamps_start.append(start)
                    timestamps_end.append(end)

        return timestamps_start, timestamps_end, speakers

    def process_rttm_file(self, path_to_annotations):
        """extract the list of timestamps_start, timestamps_end and speakers
        from an annotations file with path: path_to_annotations.

        Args:
            path_to_annotations (str): path to the annotations file.

        Returns:
            timestamps_start (list):  A list of timestamps indicating the start of each speaker segment.
            timestamps_end (list): A list of timestamps indicating the end of each speaker segment.
            speakers (list): The list of audio speakers, with their order of appearance.
        """

        timestamps_start = []
        timestamps_end = []
        speakers = []

        with open(path_to_annotations, "r") as file:
            lines = file.readlines()
            for line in lines:
                fields = line.split()

                speaker = fields[-3]
                start_time = float(fields[3])
                end_time = start_time + float(fields[4])

                timestamps_start.append(start_time)
                speakers.append(speaker)
                timestamps_end.append(end_time)

        return timestamps_start, timestamps_end, speakers

    def construct_dataset(self, num_proc=1):
        """Main method to construct the dataset

        Returns:
            self.spd_dataset: HF dataset compatible with diarizers.
        """

        self.spd_dataset = DatasetDict()

        for subset in self.audio_paths:
            timestamps_start = []
            timestamps_end = []
            speakers = []

            self.spd_dataset[str(subset)] = Dataset.from_dict({})

            for annotations in self.annotations_paths[subset]:
                if self.annotations_type == "rttm":
                    timestamps_start_file, timestamps_end_file, speakers_file = self.process_rttm_file(annotations)
                elif self.annotations_type == "cha":
                    timestamps_start_file, timestamps_end_file, speakers_file = self.process_cha_file(annotations)

                timestamps_start.append(timestamps_start_file)
                timestamps_end.append(timestamps_end_file)
                speakers.append(speakers_file)

            self.spd_dataset[subset] = Dataset.from_dict(
                {
                    "audio": self.audio_paths[subset],
                    "timestamps_start": timestamps_start,
                    "timestamps_end": timestamps_end,
                    "speakers": speakers,
                }
            ).cast_column("audio", Audio(sampling_rate=self.sample_rate))

            if self.crop_unannotated_regions:
                self.spd_dataset[subset] = (
                    self.spd_dataset[subset]
                    .map(
                        lambda example: self.crop_audio(example),
                        batched=True,
                        batch_size=8,
                        remove_columns=self.spd_dataset[subset].column_names,
                        num_proc=num_proc,
                    )
                    .cast_column("audio", Audio(sampling_rate=self.sample_rate))
                )

        return self.spd_dataset
