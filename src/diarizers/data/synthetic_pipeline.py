import copyreg
import os
import random
from itertools import chain

import numpy as np
import torch
import torchaudio.transforms as T
from audiomentations import (AddBackgroundNoise, AddGaussianSNR,
                             ApplyImpulseResponse, Compose, Gain)
from denoiser import pretrained
from denoiser.dsp import convert_audio
from tqdm import tqdm

from datasets import Audio, Dataset, concatenate_datasets, load_dataset


class SyntheticDatasetConfig:
    def __init__(
        self,
        dataset_name: str = "mozilla-foundation/common_voice_17_0",
        subset: str = "validated",
        split: str = "ja",
        speaker_column_name: str = "client_id",
        audio_column_name: str = "audio",
        min_samples_per_speaker: int = 10,
        nb_speakers_from_dataset: int = 20,
        sample_rate: int = 16000,
        num_meetings: int = 200,
        nb_speakers_per_meeting: int = 3,
        segments_per_meeting: int = 16,
        normalize: bool = True,
        augment: bool = False,
        overlap_proba: float = 0.3,
        overlap_length: float = 3,
        random_gain: bool = False,
        add_silence: bool = True,
        silence_duration: float = 3,
        silence_proba: float = 3,
        denoise: bool = False,
        bn_path: str = "/home/kamil/datasets/wham_noise/wham_noise/tr",
        ir_path: str = "/home/kamil/datasets/MIT-ir-survey",
        num_proc: int = 2,
    ) -> None:
        """_summary_

        Args:
            dataset_name (str, optional): ASR dataset to be used to generate synthetic speaker diarization meetings. Defaults to "mozilla-foundation/common_voice_17_0".
            subset (str, optional): ASR dataset subset. Defaults to 'validated'.
            split (str, optional): ASR dataset split. Defaults to "ja".
            speaker_column_name (str, optional): ASR dataset column name with speaker ids. Defaults to 'client_id'.
            audio_column_name (str, optional): ASR dataset column name with audio files. Defaults to 'audio'.
            min_samples_per_speaker (int, optional):
                Minimal number of audio samples from a given speaker in the ASR dataset
                to use the speaker for meetings generation. Defaults to 10.
            nb_speakers_from_dataset (int, optional):
                Maximum number of speakers to use from the ASR dataset to generate meetings.
                Defaults to 200.
            sample_rate (int, optional): sample rate of the generated meetings. Defaults to 16000.
            num_meetings (int, optional): number of meeting audio files to generate. Defaults to 1000.
            nb_speakers_per_meeting (int, optional): number of speakers in generated meeting. Defaults to 3.
            segments_per_meeting (int, optional): number of audio segments used in a generated meeting. Defaults to 16.
            normalize (bool, optional): normalize audio segments. Defaults to True.
            augment (bool, optional): augment generated meetings with background noise and reverberation. Defaults to False.
            overlap_proba (float, optional): Probability of adding overlap to concatenated consecutive audio segments. Defaults to 0.3.
            overlap_length (float, optional): Maximum overalp duration (in seconds) between two overlapping audio segments. Defaults to 3.
            random_gain (bool, optional): Apply random gain to each audio segments. Defaults to False.
            add_silence (bool, optional): Add silence or not in generated meeting . Defaults to True.
            silence_duration (float, optional): maximum silence duration (in seconds). Defaults to 3.
            silence_proba (float, optional): probability of adding a silence in a generated meeting. Defaults to 3.
            denoise (bool, optional): denoise the generated meeting. Defaults to False.
            bn_path (str, optional): path to background noise samples. Defaults to "/home/kamil/datasets/wham_noise/wham_noise/tr".
            ir_path (str, optional): path to impulse response samples. Defaults to "/home/kamil/datasets/MIT-ir-survey".
            num_proc (int, optional): Number of process used during generation. Defaults to 2.
        """

        # ASR dataset params:
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.min_samples_per_speaker = min_samples_per_speaker
        self.speaker_column_name = speaker_column_name
        self.audio_column_name = audio_column_name
        self.nb_speakers_from_dataset = nb_speakers_from_dataset
        self.sample_rate = sample_rate

        # Synthetic meetings meta params:
        self.num_meetings = num_meetings
        self.nb_speakers_per_meeting = nb_speakers_per_meeting
        self.segments_per_meeting = segments_per_meeting

        # Synthetic meetings properties:
        # add overlap
        self.overlap_proba = overlap_proba
        self.overlap_length = overlap_length

        # add silences:
        self.add_silence = add_silence
        self.silence_duration = silence_duration
        self.silence_proba = silence_proba

        # normalize
        self.normalize = normalize

        # Augment
        self.augment = augment
        self.bn_path = bn_path
        self.ir_path = ir_path

        # denoise:
        self.denoise = denoise

        # add random gain:
        self.random_gain = random_gain

        self.num_proc = num_proc


class SyntheticDataset:
    """Generate a synthetic speaker diarisation dataset from an ASR dataset of individual speaker audio segments."""

    def __init__(
        self,
        config=SyntheticDatasetConfig(),
    ) -> None:
        """
        Init method to the Synthetic dataset class.

        Args:
            config (SyntheticDatasetConfig): configuration
        """

        self.dataset_name = config.dataset_name
        self.subset = config.subset
        self.split = config.split
        self.min_samples_per_speaker = config.min_samples_per_speaker
        self.speaker_column_name = config.speaker_column_name
        self.audio_column_name = config.audio_column_name
        self.nb_speakers_from_dataset = config.nb_speakers_from_dataset
        self.sample_rate = config.sample_rate

        self.num_meetings = config.num_meetings
        self.nb_speakers_per_meeting = config.nb_speakers_per_meeting
        self.segments_per_meeting = config.segments_per_meeting

        self.overlap_proba = config.overlap_proba
        self.overlap_length = config.overlap_length

        self.add_silence = config.add_silence
        self.silence_duration = config.silence_duration
        self.silence_proba = config.silence_proba

        self.normalize = config.normalize

        self.augment = config.augment
        self.bn_path = config.bn_path
        self.ir_path = config.ir_path

        self.random_gain = config.random_gain

        self.denoise = config.denoise

        self.num_proc = config.num_proc

        # Load ASR dataset:
        dataset = load_dataset(str(self.dataset_name), str(self.split))
        self.dataset = dataset[str(self.subset)].select_columns(
            [str(self.speaker_column_name), str(self.audio_column_name)]
        )

        # Extract speakers and number of appearances in the dataset:
        speaker_appearance_count = self.dataset.to_pandas()[str(self.speaker_column_name)].value_counts()

        # Select only speakers with more than self.min_samples_per_speaker appearances:
        self.speakers_to_sample_from = list(
            speaker_appearance_count[speaker_appearance_count > self.min_samples_per_speaker].keys()
        )[: self.nb_speakers_from_dataset]

        print("nb speakers in dataset to keep:", len(self.speakers_to_sample_from))

        # Filter the ASR dataset to keep only samples from speakers_to_sample_from
        self.speakers_to_sample_from_dataset = self.dataset.filter(
            lambda x: x in self.speakers_to_sample_from,
            input_columns=[str(self.speaker_column_name)],
            num_proc=self.num_proc,
        )

        # Create a mapping from speaker samples --> indexes in speakers_to_sample_from_dataset
        self.speaker_indexes_in_dataset = {}
        self.speakers_to_sample_from.sort()
        self.speakers_to_sample_from_dataset = self.speakers_to_sample_from_dataset.sort("client_id")
        speaker_appearance_count = dict(speaker_appearance_count)
        index = 0
        for speaker in tqdm(self.speakers_to_sample_from):
            self.speaker_indexes_in_dataset[str(speaker)] = list(
                range(index, index + speaker_appearance_count[str(speaker)])
            )
            index = index + speaker_appearance_count[str(speaker)]

        # Define the VAD model used to refine the timestamps:
        torch.set_num_threads(1)
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True)
        self.vad_model = vad_model
        self.get_speech_timestamps = utils[0]

        # Define the denoiser:
        if self.denoise:
            self.denoiser = pretrained.dns64()

        # Define the augmentation pipeline:
        if self.augment:
            self.augmentation_pipeline = Compose(
                [
                    ApplyImpulseResponse(self.ir_path, p=0.2),
                    AddBackgroundNoise(self.bn_path, 5, 40, p=0.2),
                    AddGaussianSNR(
                        min_snr_db=30.0,
                        max_snr_db=50.0,
                        p=0.1,
                    ),
                ]
            )

        # Define Gain to apply to each segment:
        if self.random_gain:
            self.apply_gain = Gain(p=0.2, min_gain_db=-3, max_gain_db=3)

    def pickle_vad_model(self, model):
        if not os.path.exists("vad.pt"):
            model.save("vad.pt")
        return torch.jit.load, ("vad.pt",)

    def sample_next_speaker(self):
        """Next speaker sampling.

        Returns:
            self.current_speaker (str): current speaker.
        """

        other_speakers = self.sampled_speakers.copy()
        if self.current_speaker in other_speakers:
            other_speakers.remove(self.current_speaker)
        self.current_speaker = random.choice(other_speakers)

        return self.current_speaker

    def sample_meeting_segments(self):
        """Sample segments that will be used for meeting generation:

        Returns:
            batch_samples (HuggingFace dataset): batch of samples to be concatenated to from a meeting.
        """

        batch_samples = Dataset.from_dict({str(self.speaker_column_name): [], str(self.audio_column_name): []})

        # Sample nb_speakers_per_meeting from the list of speakers_to_sample_from:
        self.sampled_speakers = random.sample(self.speakers_to_sample_from, self.nb_speakers_per_meeting)
        # Get the pool of segments associated with the speakers:
        self.audio_index_pool = {
            speaker: self.speaker_indexes_in_dataset[speaker].copy() for speaker in self.sampled_speakers
        }

        self.current_speaker = self.sampled_speakers[0]

        indexes = []
        # Sample segments_per_meeting segments:
        for _ in range(self.segments_per_meeting):

            # select a segment from the current speaker and remove it from the pool of segments:
            indexes.append(random.choice(self.audio_index_pool[self.current_speaker]))
            self.audio_index_pool[self.current_speaker].remove(indexes[-1])

            if len(self.audio_index_pool[self.current_speaker]) == 0:
                del self.audio_index_pool[self.current_speaker]
                self.sampled_speakers.remove(self.current_speaker)

            # Sample next speaker
            self.current_speaker = self.sample_next_speaker()

        batch_samples = self.speakers_to_sample_from_dataset.select(indexes)

        assert len(batch_samples) == self.segments_per_meeting
        return batch_samples

    def estimate_meeting_length(self, audio_segments):
        """Estimate the audio duration of the meeting to be generated from the batch of audio segments.

        Args:
            audio_segments (list): list of audio segments.

        Returns:
            audio_duration (float): audio duration of the meeting to be generated.
        """

        audio_duration = 0

        for audio_segment in audio_segments:
            audio_duration += len(audio_segment) / self.sample_rate

        audio_duration *= 1.01

        return audio_duration

    def normalize_audio_segment(self, audio_segment):
        """Normalize audio_segment."""
        return audio_segment / max(np.max(audio_segment), -np.min(audio_segment))

    def add_gain_to_audio_segment(self, audio_segment):
        """Add gain to audio_segment"""

        audio_segment = self.apply_gain(audio_segment, sample_rate=self.sample_rate)

        return audio_segment

    def augment_audio_segment(self, audio_file):
        """Method to augment the input audio with background noise and reverb.

        Args:
            audio_file (numpy.ndarray): generated meeting audio array.

        Returns:
            audio_file (numpy.ndarray): augmented generated meeting audio array.
        """

        audio_file = self.augmentation_pipeline(samples=audio_file, sample_rate=self.sample_rate)
        return audio_file

    def refine_audio_segment_timestamps(self, audio_segment, speaker):
        """Refine audio_segment timestamps using a Voice Activity Detector.

        Args:
            audio_segment (numpy.ndarray): audio segment.
            speaker (str): speaker id.

        Returns:
            audio_segment (numpy.ndarray): croped audio segment - removes the beginning and end of the segment where there is no speech.
            file_timestamps_start (list): list with refined start timestamps.
            file_timestamps_end (list): list with refined end timestamps.
            speakers (list): List of speakers associated to file_timestamps_start and file_timestamps_end.
        """

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

    def denoise_audio_segment(self, audio_file, rank=None):
        """Method to denoise input audio.

        Args:
            audio_file (np.ndarray): generated meeting audio array.

        Returns:
            audio_file (np.ndarray): denoised generated meeting audio array.
        """

        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        self.denoiser = self.denoiser.to(device)

        audio_file_converted = convert_audio(
            torch.tensor(audio_file).unsqueeze(0).to(device),
            self.sample_rate,
            self.denoiser.sample_rate,
            self.denoiser.chin,
        )
        with torch.no_grad():
            audio_file = (
                self.denoiser(torch.tensor(audio_file_converted, dtype=torch.float32))[0].squeeze(0).cpu().numpy()
            )
        return audio_file

    def add_silences_to_audio_segment(
        self,
        audio_file,
        file_timestamps_start,
        file_timestamps_end,
    ):
        """Randomly add silences to generated meeting arrays.

        Args:
            audio_file (np.ndarray): generated meeting audio array.
            file_timestamps_start (list): list of meeting level start timestamps.
            file_timestamps_end (list): list of meeting level end timestamps.

        Returns:
            extended_audio_file (np.ndarray): Updated generated meeting audio array (possibly with silences).
            file_timestamps_start (list): updated list of start timestamps.
            file_timestamps_end (list): updated list of end timestamps.
        """

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

    def create_meeting(
        self,
        audio_segments,
        file_timestamps_start_vad,
        file_timestamps_end_vad,
        speakers_vad,
    ):
        """generate the multi speakers audio array and associated timestamps.

        Args:
            audio_segments (list): list of audio segments to be concatenated
            file_timestamps_start_vad (list): list of list of start timestamps.
            file_timestamps_end_vad (list): list of list of end timestamps.
            speakers_vad (list): list of list with speaker ids.

        Returns:
            audio_file (np.ndarray): generated meeting audio array.
            file_timestamps_start: list of meeting level start timestamps
            file_timestamps_end: list of meeting level end timestamps.
            speakers: list of meeting level speakers.
        """

        start = 0
        # Estimate the audio duration of the meeting to be generated:
        audio_duration = self.estimate_meeting_length(audio_segments)
        audio_file = np.zeros(int(audio_duration * self.sample_rate))
        audio_file_length = len(audio_file)

        # Meeting level timestamps and speakers:
        file_timestamps_start = []
        file_timestamps_end = []
        speakers = []

        is_overlap = False
        for i, audio_segment in enumerate(audio_segments):

            start_index = int(start * self.sample_rate)

            if start_index >= audio_file_length:
                break

            segment_length = min(audio_file_length - start_index, len(audio_segment))

            # Concatenate audio segments:
            audio_file[start_index : start_index + segment_length] += audio_segment[:segment_length]

            # Update the meeting level timestamps and speaker lists:
            file_timestamps_start.append(
                [timestamps_start + start for timestamps_start in file_timestamps_start_vad[i]]
            )
            file_timestamps_end.append([timestamps_end + start for timestamps_end in file_timestamps_end_vad[i]])
            speakers.append(speakers_vad[i])

            end = start + len(audio_segment) / self.sample_rate

            # Sample the next start position from a rayleight distribution with 200ms mode to model natural human conversation:
            if np.random.rand() < self.overlap_proba and not is_overlap:
                start = max(0, end + np.random.rayleigh(0.002) - 0.002 - self.overlap_length * np.random.rand())
                is_overlap = True  # We add this to make sure we don't apply overlap to multiple successive samples
            else:
                start = max(0, end + np.random.rayleigh(0.002) - 0.002)
                is_overlap = False

        file_timestamps_start = list(chain.from_iterable(file_timestamps_start))
        file_timestamps_end = list(chain.from_iterable(file_timestamps_end))
        speakers = list(chain.from_iterable(speakers))

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
        rank,
    ):
        """Concatenate a batch of audio segments to form a synthetic meeting audio file, to be used with a HF .map function

        Args:
            file (dict): dataset files with "audio" feature.
            rank (_type_): _description_

        Returns:
            new_batch: new batch containing the generated audio meeting file with timestamps and speakers.
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

            if self.random_gain:
                audio_segment = self.add_gain_to_audio_segment(audio_segment)

            # Refine segment level timestamps:
            (
                audio_segment,
                timestamps_start_vad,
                timestamps_end_vad,
                speakers_vad,
            ) = self.refine_audio_segment_timestamps(
                audio_segment,
                element["client_id"],
            )
            file_timestamps_start.append(timestamps_start_vad)
            file_timestamps_end.append(timestamps_end_vad)
            speakers.append(speakers_vad)
            audio_segments.append(audio_segment)

        (audio_file, file_timestamps_start, file_timestamps_end, speakers) = self.create_meeting(
            audio_segments, file_timestamps_start, file_timestamps_end, speakers
        )

        if self.add_silence:
            (
                audio_file,
                file_timestamps_start,
                file_timestamps_end,
            ) = self.add_silences_to_audio_segment(audio_file, file_timestamps_start, file_timestamps_end)

        if self.denoise:
            audio_file = self.denoise_audio_segment(audio_file, rank=rank)

        if self.augment:
            audio_file = self.augment_audio_segment(audio_file)

        if self.normalize:
            audio_file = self.normalize_audio_segment(audio_file)

        audio_file = {
            "array": np.array(audio_file),
            "sampling_rate": self.sample_rate,
        }

        new_batch["speakers"].append(speakers)
        new_batch["audio"].append(audio_file)
        new_batch["timestamps_start"].append(file_timestamps_start)
        new_batch["timestamps_end"].append(file_timestamps_end)

        return new_batch

    def generate(
        self,
    ):
        """Main method to generate a speaker diarization synthetic dataset.

        Returns:
            final_dataset (Hugging Face datasets): final synthetic dataset.
        """

        if self.denoise:
            # For the moment, the demucs denoiser doesn't support multi-threading.
            self.num_proc = 1

        if self.num_proc > 1:
            copyreg.pickle(type(self.vad_model), self.pickle_vad_model)

        audio_samples = Dataset.from_dict({str(self.speaker_column_name): [], str(self.audio_column_name): []})

        if self.num_proc > 1:
            # Do this to force all batches to have the same size when num_proc > 1:
            self.num_meetings = (self.num_meetings // self.num_proc) * self.num_proc

        # Select samples to be used in each synthetic meeting and save them in audio_samples dataset:
        for _ in tqdm(range(self.num_meetings)):

            meeting_samples = self.sample_meeting_segments()
            audio_samples = concatenate_datasets([audio_samples, meeting_samples])

        # Concatenate the selected audio segments to form meetings:
        final_dataset = audio_samples.map(
            self.concatenate,
            batched=True,
            batch_size=self.segments_per_meeting,
            remove_columns=audio_samples.column_names,
            with_rank=True if torch.cuda.device_count() > 0 else False,
            num_proc=self.num_proc,
        ).cast_column("audio", Audio(sampling_rate=self.sample_rate))

        if self.num_proc > 1:
            copyreg.dispatch_table.pop(type(self.vad_model), None)
            if os.path.exists("vad.pt"):
                os.remove("vad.pt")

        return final_dataset


if __name__ == "__main__":

    synthetic_config = SyntheticDatasetConfig(num_proc=1)
    synthetic_dataset = SyntheticDataset(synthetic_config)
    synthetic_dataset.generate()
