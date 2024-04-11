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
from datasets import Dataset, DatasetDict, concatenate_datasets, Audio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import copyreg
import os
from itertools import chain
import bisect



torch.multiprocessing.set_start_method('spawn')

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

        self.num_samples = config["num_samples"]
        self.audio_file_length = config["audio_file_length"]
        self.batch_size = config["batch_size"]
        self.std_concatenate = config["std_concatenate"]
        self.sample_rate = config["sample_rate"]
        self.denoise = config["denoise"]
        self.normalize = config["normalize"]
        self.augment = config["augment"]
        self.silent_regions = config["silent_regions"]["silent_regions"]
        self.silence_duration = config["silent_regions"]["silence_duration"]
        self.silence_proba = config["silent_regions"]["silence_proba"]
        self.short_audio_threshold = config["short_audio_threshold"]

        if self.denoise:
            self.denoiser = pretrained.dns64().cuda()

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
                    ApplyImpulseResponse(self.ir_path, p=0.5),
                    AddBackgroundNoise(self.bn_path, 20, 60, p=0.5),
                    AddGaussianSNR(
                        min_snr_db=30.0,
                        max_snr_db=50.0,
                        p=0.1,
                    ),
                ]
            )

    def estimate_concat_audio_length(self, audio_segments, threshold = None):
        """_summary_

        Args:
            batch (_type_): _description_
            sr (_type_): _description_

        Returns:
            _type_: _description_
        """

        audio_duration = 0
         
        if threshold != None: 
            self.short_audio_threshold = threshold 
        else: 
            self.short_audio_threshold = sorted([len(audio_segment) / self.sample_rate for audio_segment in audio_segments])[0]

        for audio_segment in audio_segments:
            duration = len(audio_segment) / self.sample_rate
            if duration > self.short_audio_threshold: 
                audio_duration += len(audio_segment) / self.sample_rate

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

    def refine_timestamps(self, audio_segment, speaker):

        speech_timestamps = self.get_speech_timestamps(
            audio_segment, self.vad_model, sampling_rate=self.sample_rate
        )

        if len(speech_timestamps): 
            audio_segment_start_index = int(speech_timestamps[0]['start'])
            audio_segment_end_index = int(speech_timestamps[-1]['end'])
            audio_segment = audio_segment[audio_segment_start_index:audio_segment_end_index]

            file_timestamps_start = [
                (timestamps["start"]- speech_timestamps[0]['start'])/ self.sample_rate
                for timestamps in speech_timestamps
            ]
            file_timestamps_end = [
                (timestamps["end"]- speech_timestamps[0]['start']) / self.sample_rate
                for timestamps in speech_timestamps
            ]
            speakers = [speaker] * len(speech_timestamps)

        else: 
            file_timestamps_start = [0]
            file_timestamps_end = [len(audio_segment)/self.sample_rate]
            speakers = [speaker]

        assert len(speakers) > 0

        return (audio_segment, file_timestamps_start, file_timestamps_end, speakers)

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

    def insert_audio_segments(
        self, 
        audio_segments, 
        file_timestamps_start_vad, 
        file_timestamps_end_vad, 
        speakers_vad, 
        threshold = None, 
    ): 

        start = 0
        audio_duration = self.estimate_concat_audio_length(audio_segments, threshold)
        audio_file = np.zeros(int(audio_duration * self.sample_rate))
        audio_file_length = len(audio_file)

        short_audios_segments = []

        file_timestamps_start_long = []
        file_timestamps_end_long = []
        speakers_long = []
        segments_durations_long = []

        speakers_short = []

        for i, audio_segment in enumerate(audio_segments): 
            
            if len(audio_segment) / self.sample_rate > self.short_audio_threshold: 
                
                start_index = int(start * self.sample_rate)

                if start_index >= audio_file_length:
                    break

                segment_length = min(audio_file_length - start_index, len(audio_segment))

                audio_file[start_index : start_index + segment_length] += audio_segment[
                    :segment_length
                ]

                file_timestamps_start_long.append([timestamps_start + start for  timestamps_start in file_timestamps_start_vad[i]])
                file_timestamps_end_long.append([timestamps_end + start for  timestamps_end in file_timestamps_end_vad[i]])
                segments_durations_long.append(len(audio_segment) / self.sample_rate)
                speakers_long.append(speakers_vad[i])
              
                end = start + len(audio_segment) / self.sample_rate
                start = end + np.random.rayleigh(0.002) - 0.002
            else: 
                short_audios_segments.append(audio_segment)
                speakers_short.append(speakers_vad[i])

        sorted_indexes = [e[0] for e in sorted(enumerate(segments_durations_long), key=lambda x: x[1], reverse=True)]
        
        if len(short_audios_segments) > 0: 
            for i in range(len(short_audios_segments)): 

                if i >= len(sorted_indexes): 
                    break
                
                short_audio_segment = short_audios_segments[i]
                short_audio_duration = (len(short_audio_segment)/self.sample_rate)
                start = file_timestamps_start_long[sorted_indexes[i]][0]
                end = max(file_timestamps_end_long[sorted_indexes[i]])

                assert short_audio_duration < end-start

                short_audio_start = np.random.uniform(start, end - short_audio_duration)
                insert_index_start = int(short_audio_start * self.sample_rate)
                segment_length = int(short_audio_duration * self.sample_rate)

                short_audio_end = short_audio_start + short_audio_duration

                audio_file[insert_index_start : insert_index_start + segment_length] += short_audio_segment[
                        :segment_length
                    ]
                
                index = bisect.bisect_left(file_timestamps_start_long[sorted_indexes[i]], short_audio_start)

                file_timestamps_start_long[sorted_indexes[i]].insert(index, short_audio_start)
                file_timestamps_end_long[sorted_indexes[i]].insert(index, short_audio_end)
                speakers_long[sorted_indexes[i]].insert(index, speakers_short[i][0])

        file_timestamps_start = list(chain.from_iterable(file_timestamps_start_long))
        file_timestamps_end = list(chain.from_iterable(file_timestamps_end_long))
        speakers = list(chain.from_iterable(speakers_long))

        file_timestamps_start = [min(timestamp_start, len(audio_file)/ self.sample_rate) for timestamp_start in file_timestamps_start]
        file_timestamps_end = [min(timestamp_end, len(audio_file)/ self.sample_rate) for timestamp_end in file_timestamps_end]

        return audio_file, file_timestamps_start, file_timestamps_end, speakers


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

        file_timestamps_start = []
        file_timestamps_end = []
        speakers = []
        audio_segments = []

        for element in batch:

            audio_segment = element["audio"]["array"]

            resample = T.Resample(sr, self.sample_rate)
            audio_segment = (
                resample(torch.tensor(audio_segment, dtype=torch.float32))
                .cpu()
                .numpy()
            )

            if self.normalize:
                audio_segment = self.normalize_audio(audio_segment)
            
            (
                audio_segment, 
                timestamps_start_vad,
                timestamps_end_vad,
                speakers_vad,
            ) = self.refine_timestamps(
                audio_segment,
                element["client_id"],
            )
            file_timestamps_start.append(timestamps_start_vad)
            file_timestamps_end.append(timestamps_end_vad)
            speakers.append(speakers_vad)
            audio_segments.append(audio_segment)
        
        # Could happen that len(audio_segments)!= self.batch_size when proc > 1:
        # if len(audio_segments) == self.batch_size: 
        (
            audio_file, 
            file_timestamps_start, 
            file_timestamps_end, 
            speakers
        ) = self.insert_audio_segments(
            audio_segments, file_timestamps_start, file_timestamps_end, speakers
        )
        # else:
        #     ## In that case, we don't have enough 
        #     (
        #         audio_file, 
        #         file_timestamps_start, 
        #         file_timestamps_end, 
        #         speakers
        #     ) = self.insert_audio_segments(
        #         audio_segments, file_timestamps_start, file_timestamps_end, speakers, threshold=0
        #     )

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

        if num_proc>1: 
            # serialize vad_model to allow multiprocessing in the map function
            copyreg.pickle(type(self.vad_model), pickle_model)

        self.input_dataset.select_columns([str(self.speaker_column_name), str(self.audio_column_name)])

        for subset in subsets:

            concatenate_dataset = Dataset.from_dict(
                {"audio": [], "speakers": [], "timestamps_start": [], "timestamps_end": []}
            )

            if subset == 'train': 
                num_samples = self.num_samples
            if subset in ['validation', 'test']: 
                num_samples = int(0.2 * self.num_samples)
            
            if num_proc>1:
                # Do this to force all batches to have the same size when num_proc > 1: 
                num_samples = (num_samples // num_proc) * num_proc

            nb_samples = min(num_samples * self.batch_size, len(self.input_dataset[str(subset)]))
            
            dataset = self.input_dataset[str(subset)].shuffle().select(range(nb_samples))

            result = dataset.map(
                lambda example: self.concatenate(example),
                batched=True,
                batch_size=self.batch_size,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
            ).cast_column("audio", Audio(sampling_rate=self.sample_rate))

            dataset = concatenate_datasets([concatenate_dataset, result])

            self.spd_dataset[str(subset)] = dataset

        if num_proc>1: 
            copyreg.dispatch_table.pop(type(self.vad_model), None)
            if os.path.exists("vad.pt"):
                os.remove("vad.pt")

        return self.spd_dataset
