import random

import numpy as np
import torch
import torchaudio.transforms as T
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianSNR,
    ApplyImpulseResponse,
    Compose,
)
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from denoiser import pretrained
from denoiser.dsp import convert_audio
import copyreg
import os


def pickle_model(model):
    if not os.path.exists("vad.pt"):
        model.save("vad.pt")
    return torch.jit.load, ("vad.pt",)

class SyntheticDataset:
    """_summary_"""

    def __init__(
        self,
        input_dataset,
        speaker_column_name,
        audio_column_name,
        config,
    ):
        self.input_dataset = input_dataset
        self.speaker_column_name = speaker_column_name
        self.audio_column_name = audio_column_name

        self.audio_file_length = config["audio_file_length"]
        self.batch_size = config["batch_size"]
        self.std_concatenate = config["std_concatenate"]
        self.sample_rate = config["sample_rate"]
        self.refine_with_vad = config["refine_with_vad"]
        self.denoise = config["denoise"]
        self.normalize = config["normalize"]
        self.augment = config["augment"]
        self.silent_regions = config["silent_regions"]["silent_regions"]
        self.silence_duration = config["silent_regions"]["silence_duration"]
        self.silence_proba = config["silent_regions"]["silence_proba"]

        if self.denoise:
            self.denoiser = pretrained.dns64().cuda()

        if self.refine_with_vad:

            torch.set_num_threads(1)
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
            )
            self.vad_model = vad_model
            self.get_speech_timestamps = utils[0]

        if self.augment:

            self.bn_path = config["bn_path"]
            self.ir_path = config["ir_path"]

            self.augmentation_pipeline = Compose(
                [
                    ApplyImpulseResponse(self.ir_path, p=0.3),
                    AddBackgroundNoise(self.bn_path, 30, 50, p=0.1),
                    AddGaussianSNR(
                        min_snr_db=30.0,
                        max_snr_db=50.0,
                        p=0.2,
                    ),
                ]
            )

    def estimate_audio_duration(self, batch, sr):
        """_summary_

        Args:
            batch (_type_): _description_
            sr (_type_): _description_

        Returns:
            _type_: _description_
        """

        audio_duration = 0
        for row in batch:
            audio_duration += len(row["audio"]["array"]) / sr

        audio_duration *= self.audio_file_length

        return audio_duration

    def normalize_audio(self, audio_segment):
        """_summary_

        Args:
            audio_segment (_type_): _description_

        Returns:
            _type_: _description_
        """

        return audio_segment / max(np.max(audio_segment), -np.min(audio_segment))

    def denoise_audio(self, audio_file):
        """_summary_

        Args:
            audio_file (_type_): _description_

        Returns:
            _type_: _description_
        """

        audio_file_converted = convert_audio(
            torch.tensor(audio_file).unsqueeze(0).cuda(),
            self.sample_rate,
            self.denoiser.sample_rate,
            self.denoiser.chin,
        )
        with torch.no_grad():
            audio_file = (
                self.denoiser(torch.tensor(audio_file_converted, dtype=torch.float32))[
                    0
                ]
                .squeeze(0)
                .cpu()
                .numpy()
            )

        return audio_file

    def augment_audio(self, audio_file):
        """_summary_

        Args:
            audio_file (_type_): _description_

        Returns:
            _type_: _description_
        """

        audio_file = self.augmentation_pipeline(
            samples=audio_file, sample_rate=self.sample_rate
        )
        return audio_file

    def refine_timestamps(self, audio_segment, speaker, start):

        speech_timestamps = self.get_speech_timestamps(
            audio_segment, self.vad_model, sampling_rate=self.sample_rate
        )

        file_timestamps_start = [
            start + timestamps["start"] / self.sample_rate
            for timestamps in speech_timestamps
        ]
        file_timestamps_end = [
            start + timestamps["end"] / self.sample_rate
            for timestamps in speech_timestamps
        ]
        speakers = [speaker] * len(speech_timestamps)

        return (file_timestamps_start, file_timestamps_end, speakers)

    def add_silent_regions(
        self,
        audio_file,
        file_timestamps_start,
        file_timestamps_end,
    ):

        if random.random() < self.silence_proba and len(file_timestamps_start) > 2:
            duration = np.maximum(np.random.normal(self.silence_duration, 3.0), 1)

            insert_silence_index = random.randint(0, len(file_timestamps_start) - 2)

            silence_start = file_timestamps_end[insert_silence_index]
            silence_end = silence_start + duration
            silence_start_index = int(silence_start * self.sample_rate)
            silence_end_index = int(silence_end * self.sample_rate)

            relative_duration = silence_end - min(
                file_timestamps_start[insert_silence_index + 1 :]
            )
            file_timestamps_start[insert_silence_index + 1 :] += relative_duration
            file_timestamps_end[insert_silence_index + 1 :] += relative_duration

            new_length = int(relative_duration * self.sample_rate) + len(audio_file)
            extended_audio_file = np.zeros(new_length)

            extended_audio_file[:silence_start_index] = audio_file[:silence_start_index]

            length_segment_end = max(1, len(extended_audio_file[silence_end_index:]))

            extended_audio_file[-length_segment_end:] = audio_file[-length_segment_end:]

        else:
            extended_audio_file = audio_file

        return extended_audio_file, file_timestamps_start, file_timestamps_end

    def concatenate(
        self,
        files,
    ):
        """_summary_

        Args:
            files (_type_): _description_

        Returns:
            _type_: _description_
        """

        new_batch = {
            "audio": [],
            "speakers": [],
            "timestamps_start": [],
            "timestamps_end": [],
        }

        sr = files["audio"][0]["sampling_rate"]

        batch = [
            {key: values[i] for key, values in files.items()}
            for i in range(len(files["audio"]))
        ]

        audio_duration = self.estimate_audio_duration(batch, sr)
        audio_file = np.zeros(int(audio_duration * self.sample_rate))
        audio_file_length = len(audio_file)

        start = 0

        file_timestamps_start = []
        file_timestamps_end = []
        speakers = []

        for element in batch:

            audio_segment = element["audio"]["array"]

            if self.sample_rate:
                resample = T.Resample(sr, self.sample_rate)
                audio_segment = (
                    resample(torch.tensor(audio_segment, dtype=torch.float32))
                    .cpu()
                    .numpy()
                )

            if self.normalize:
                audio_segment = self.normalize_audio(audio_segment)

            dur = len(audio_segment) / self.sample_rate
            end = start + dur

            start_index = int(start * self.sample_rate)

            if start_index >= audio_file_length:
                break

            segment_length = min(audio_file_length - start_index, len(audio_segment))

            if self.refine_with_vad:
                (
                    file_timestamps_start_vad,
                    file_timestamps_end_vad,
                    speakers_vad,
                ) = self.refine_timestamps(
                    audio_segment,
                    element["client_id"],
                    start,
                )
                file_timestamps_start += file_timestamps_start_vad
                file_timestamps_end += file_timestamps_end_vad
                speakers += speakers_vad

            else:
                file_timestamps_start.append(start)
                file_timestamps_end.append(end)
                speakers.append(element["client_id"])

            audio_file[start_index : start_index + segment_length] += audio_segment[
                :segment_length
            ]
            start = max(int(0), np.random.normal(end, self.std_concatenate))

        if self.silent_regions:
            (
                audio_file,
                file_timestamps_start,
                file_timestamps_end,
            ) = self.add_silent_regions(
                audio_file, file_timestamps_start, file_timestamps_end
            )

        if self.denoise:
            audio_file = self.denoise_audio(audio_file)

        if self.augment:
            audio_file = self.augment_audio(audio_file)

        if self.normalize:
            audio_file = self.normalize_audio(audio_file)

        audio_file = {
            "array": np.array(audio_file),
            "sampling_rate": self.sample_rate,
        }

        new_batch["speakers"].append(speakers)
        new_batch["audio"].append(audio_file)
        new_batch["timestamps_start"].append(file_timestamps_start)
        new_batch["timestamps_end"].append(file_timestamps_end)

        return new_batch


    def create_spd_dataset(
        self, 
        num_proc=1, 
    ):
        """_summary_

        Args:
            asr_dataset (_type_): _description_
            speaker_column_name (_type_): _description_
            audio_column_name (_type_): _description_
            config (_type_): _description_
            batch_size (_type_): _description_
            num_proc (int, optional): _description_. Defaults to 12.

        Returns:
            _type_: _description_
        """

        subsets = ["train", "validation", "test"]

        self.spd_dataset = DatasetDict(
            {
                "train": Dataset.from_dict({}),
                "validation": Dataset.from_dict({}),
                "test": Dataset.from_dict({}),
            }
        )

        if num_proc >1: 
            ## serialize vad_model to allow multiprocessing in the map function
            copyreg.pickle(type(self.vad_model), pickle_model)

        self.input_dataset.select_columns([str(self.speaker_column_name), str(self.audio_column_name)])

        for subset in subsets:

            concatenate_dataset = Dataset.from_dict(
                {"audio": [], "speakers": [], "timestamps_start": [], "timestamps_end": []}
            )

            # asr_dataset[str(subset)] = asr_dataset[str(subset)].shuffle().select(range(30))
            # speakers = set(asr_dataset[str(subset)]["client_id"])

            # while len(speakers) > 5:
            # n_speakers = random.randint(3, 10)
            # sampled_speakers = random.sample(speakers, min(n_speakers, len(speakers)))

            # dataset = asr_dataset[str(subset)].filter(
            #                 lambda x: x in sampled_speakers, input_columns=['client_id']
            #             )
            # speakers.difference_update(set(sampled_speakers))

            dataset = self.input_dataset[str(subset)].shuffle()

            result = dataset.map(
                lambda example: self.concatenate(example),
                batched=True,
                batch_size=self.batch_size,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
            )

            concatenate_dataset = concatenate_datasets([concatenate_dataset, result])

            self.spd_dataset[str(subset)] = concatenate_dataset

        if num_proc >1: 
            copyreg.dispatch_table.pop(type(self.vad_model), None)
            if os.path.exists("vad.pt"):
                os.remove("vad.pt")

        return self.spd_dataset
