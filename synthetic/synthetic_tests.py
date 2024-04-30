import copyreg
import os
import random
from itertools import chain

import numpy as np
import torch
import torchaudio.transforms as T
from audiomentations import AddBackgroundNoise, AddGaussianSNR, ApplyImpulseResponse, Compose

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset

torch.multiprocessing.set_start_method("spawn")


def pickle_model(model):
    if not os.path.exists("vad.pt"):
        model.save("vad.pt")
    return torch.jit.load, ("vad.pt",)


class SyntheticDataset:
    def __init__(
        self,
        dataset_name="mozilla-foundation/common_voice_16_1",
        split="en",
        subset="train",
        speaker_column_name="client_id",
        audio_column_name="audio",
        min_samples_per_speaker=10,
        nb_speakers_per_meeting=3,
        num_meetings=200,
        batch_size=32,
        probability_same=0.1,
        num_proc=24,
        sample_rate=16000,
    ) -> None:

        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.min_samples_per_speaker = min_samples_per_speaker
        self.speaker_column_name = speaker_column_name
        self.audio_column_name = audio_column_name
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.nb_speakers_per_meeting = nb_speakers_per_meeting
        self.num_meetings = num_meetings
        self.probability_same = probability_same
        self.sample_rate = sample_rate

        self.normalize = True
        self.augment = False

        self.silent_regions = False
        self.silence_duration = 5
        self.silence_proba = 0.8

        dataset = load_dataset(str(self.dataset_name), str(self.split))

        result = dataset[str(self.subset)].to_pandas()[str(self.speaker_column_name)].value_counts()

        # Sample only from speakers with more than 10 samples:
        self.speakers_to_sample_from = list(result[result > min_samples_per_speaker].keys())

        # Filter the dataset to keep only potential speaker candidates:
        dataset = dataset[str(self.subset)].filter(
            lambda x: x in self.speakers_to_sample_from,
            input_columns=[str(self.speaker_column_name)],
            num_proc=self.num_proc,
        )
        self.dataset = dataset.select_columns([str(self.speaker_column_name), str(self.audio_column_name)])

        torch.set_num_threads(1)
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True)
        self.vad_model = vad_model
        self.get_speech_timestamps = utils[0]

        if self.augment:
            self.bn_path = "/home/kamil/datasets/MIT-ir-survey"
            self.ir_path = "/home/kamil/datasets/wham_noise/wham_noise/tr"

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

    def sample_next_speaker(self):

        if random.random() < self.probability_same:
            self.current_speaker = self.current_speaker
        else:
            other_speakers = self.sampled_speakers.copy()
            other_speakers.remove(self.current_speaker)
            self.current_speaker = random.choice(other_speakers)

        return self.current_speaker

    def generate_pool(self):

        self.sampled_speakers = random.sample(self.speakers_to_sample_from, self.nb_speakers_per_meeting)

        # Generate the pool strategy:
        self.audio_pool = {}
        for speaker in self.sampled_speakers:
            self.audio_pool[str(speaker)] = self.dataset.filter(
                lambda x: x in speaker, input_columns=[str(self.speaker_column_name)], num_proc=self.num_proc
            ).shuffle()

    def sample_from_pool(self):

        batch_samples = Dataset.from_dict(
            {
                "speakers": [],
                "audio": [],
            }
        )

        self.current_speaker = self.sampled_speakers[0]
        sample = self.audio_pool[self.current_speaker].select(range(1))

        iterations = 0
        while iterations < self.batch_size:

            # Remove already used samples from the pool of candidates:
            self.audio_pool[self.current_speaker] = self.audio_pool[self.current_speaker].select(
                (i for i in range(1, len(self.audio_pool[self.current_speaker])))
            )

            batch_samples = concatenate_datasets([batch_samples, sample])
            self.current_speaker = self.sample_next_speaker()

            sample = self.audio_pool[self.current_speaker].select(range(1))

            iterations += 1

        return batch_samples

    def estimate_concat_audio_length(self, audio_segments):
        """_summary_

        Args:
            batch (_type_): _description_
            sr (_type_): _description_

        Returns:
            _type_: _description_
        """

        audio_duration = 0

        for audio_segment in audio_segments:
            audio_duration += len(audio_segment) / self.sample_rate

        audio_duration *= 1.1

        return audio_duration

    def normalize_audio(self, audio_segment):
        """_summary_

        Args:
            audio_segment (_type_): _description_

        Returns:
            _type_: _description_
        """

        return audio_segment / max(np.max(audio_segment), -np.min(audio_segment))

    def augment_audio(self, audio_file):
        """_summary_

        Args:
            audio_file (_type_): _description_

        Returns:
            _type_: _description_
        """

        audio_file = self.augmentation_pipeline(samples=audio_file, sample_rate=self.sample_rate)
        return audio_file

    def refine_timestamps(self, audio_segment, speaker):

        speech_timestamps = self.get_speech_timestamps(audio_segment, self.vad_model, sampling_rate=self.sample_rate)

        if len(speech_timestamps):
            audio_segment_start_index = int(speech_timestamps[0]["start"])
            audio_segment_end_index = int(speech_timestamps[-1]["end"])
            audio_segment = audio_segment[audio_segment_start_index:audio_segment_end_index]

            file_timestamps_start = [
                (timestamps["start"] - speech_timestamps[0]["start"]) / self.sample_rate
                for timestamps in speech_timestamps
            ]
            file_timestamps_end = [
                (timestamps["end"] - speech_timestamps[0]["start"]) / self.sample_rate
                for timestamps in speech_timestamps
            ]
            speakers = [speaker] * len(speech_timestamps)

        else:
            file_timestamps_start = [0]
            file_timestamps_end = [len(audio_segment) / self.sample_rate]
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

            relative_duration = silence_end - min(file_timestamps_start[insert_silence_index + 1 :])
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

    def create_multi_speakers_audio(
        self,
        audio_segments,
        file_timestamps_start_vad,
        file_timestamps_end_vad,
        speakers_vad,
    ):

        start = 0
        audio_duration = self.estimate_concat_audio_length(audio_segments)
        audio_file = np.zeros(int(audio_duration * self.sample_rate))
        audio_file_length = len(audio_file)

        file_timestamps_start_long = []
        file_timestamps_end_long = []
        speakers_long = []
        segments_durations_long = []

        for i, audio_segment in enumerate(audio_segments):

            start_index = int(start * self.sample_rate)

            if start_index >= audio_file_length:
                break

            segment_length = min(audio_file_length - start_index, len(audio_segment))

            audio_file[start_index : start_index + segment_length] += audio_segment[:segment_length]

            file_timestamps_start_long.append(
                [timestamps_start + start for timestamps_start in file_timestamps_start_vad[i]]
            )
            file_timestamps_end_long.append([timestamps_end + start for timestamps_end in file_timestamps_end_vad[i]])
            segments_durations_long.append(len(audio_segment) / self.sample_rate)
            speakers_long.append(speakers_vad[i])

            end = start + len(audio_segment) / self.sample_rate
            start = end + np.random.rayleigh(0.002) - 0.002

        file_timestamps_start = list(chain.from_iterable(file_timestamps_start_long))
        file_timestamps_end = list(chain.from_iterable(file_timestamps_end_long))
        speakers = list(chain.from_iterable(speakers_long))

        file_timestamps_start = [
            min(timestamp_start, len(audio_file) / self.sample_rate) for timestamp_start in file_timestamps_start
        ]
        file_timestamps_end = [
            min(timestamp_end, len(audio_file) / self.sample_rate) for timestamp_end in file_timestamps_end
        ]

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

        batch = [{key: values[i] for key, values in files.items()} for i in range(len(files["audio"]))]

        file_timestamps_start = []
        file_timestamps_end = []
        speakers = []
        audio_segments = []

        for element in batch:

            audio_segment = element["audio"]["array"]

            resample = T.Resample(sr, self.sample_rate)
            audio_segment = resample(torch.tensor(audio_segment, dtype=torch.float32)).cpu().numpy()

            if self.normalize:
                audio_segment = self.normalize_audio(audio_segment)

            (audio_segment, timestamps_start_vad, timestamps_end_vad, speakers_vad,) = self.refine_timestamps(
                audio_segment,
                element["client_id"],
            )
            file_timestamps_start.append(timestamps_start_vad)
            file_timestamps_end.append(timestamps_end_vad)
            speakers.append(speakers_vad)
            audio_segments.append(audio_segment)

        (audio_file, file_timestamps_start, file_timestamps_end, speakers) = self.create_multi_speakers_audio(
            audio_segments, file_timestamps_start, file_timestamps_end, speakers
        )

        if self.silent_regions:
            (
                audio_file,
                file_timestamps_start,
                file_timestamps_end,
            ) = self.add_silent_regions(audio_file, file_timestamps_start, file_timestamps_end)

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
    ):

        if self.num_proc > 1:
            copyreg.pickle(type(self.vad_model), pickle_model)

        audio_samples = Dataset.from_dict({str(self.speaker_column_name): [], str(self.audio_column_name): []})

        if self.num_proc > 1:
            # Do this to force all batches to have the same size when num_proc > 1:
            self.num_meetings = (self.num_meetings // self.num_proc) * self.num_proc

        for i in range(self.num_meetings):

            self.generate_pool()
            batch_samples = self.sample_from_pool()
            audio_samples = concatenate_datasets([audio_samples, batch_samples])

        final_dataset = audio_samples.map(
            lambda example: self.concatenate(example),
            batched=True,
            batch_size=self.batch_size,
            remove_columns=batch_samples.column_names,
            num_proc=self.num_proc,
        ).cast_column("audio", Audio(sampling_rate=self.sample_rate))

        if self.num_proc > 1:
            copyreg.dispatch_table.pop(type(self.vad_model), None)
            if os.path.exists("vad.pt"):
                os.remove("vad.pt")

        return final_dataset


if __name__ == "__main__":

    synthetic_dataset = SyntheticDataset(
        num_proc=12,
        num_meetings=20,
        nb_speakers_per_meeting=3,
        batch_size=32,
    ).create_spd_dataset()

    synthetic_dataset.push_to_hub("kamilakesbi/synthetic_dataset_en")
