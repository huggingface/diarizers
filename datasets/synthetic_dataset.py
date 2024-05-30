from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser

from diarizers import SyntheticDataset, SyntheticDatasetConfig


@dataclass
class ASRDatasetArguments:

    dataset_name: str = field(
        default="mozilla-foundation/common_voice_17_0",
        metadata={
            "help": "name of the ASR dataset to be used to generate synthetic meetings. Defaults to 'mozilla-foundation/common_voice_17_0'"
        },
    )
    subset: str = field(default="validated", metadata={"help": "ASR dataset subset. Defaults to 'validated'."})

    split: str = field(default="ja", metadata={"help": "ASR dataset split. Defaults to 'ja'"})

    speaker_column_name: str = field(
        default="client_id",
        metadata={"help": "Speaker column name. Default to 'client_id'"},
    )

    audio_column_name: str = field(
        default="audio",
        metadata={"help": "Audio column name. Default to 'audio'"},
    )

    min_samples_per_speaker: int = field(
        default=10,
        metadata={
            "help": "Minimal number of audio samples associated to a given speaker in the ASR dataset to use him in synthetic meeting generation. Defaults to 10."
        },
    )

    nb_speakers_from_dataset: int = field(
        default=-1,
        metadata={
            "help": "Number of speakers to keep for synthetic meeting generation. The speakers with the highest number of audio segments will be kept.Default to -1"
        },
    )

    sample_rate: int = field(
        default=16000, metadata={"help": "sample rate of the generated meetings. Defaults to 16000."}
    )


@dataclass
class SyntheticMeetingArguments:

    nb_speakers_per_meeting: int = field(
        default=3, metadata={"help": "number of speakers in generated meeting. Defaults to 3."}
    )
    num_meetings: int = field(
        default=1600, metadata={"help": "Number of meeting audio files to generate. Defaults to 1600."}
    )

    segments_per_meeting: int = field(
        default=16, metadata={"help": "number of audio segments used in a generated meeting. Defaults to 16."}
    )

    normalize: bool = field(
        default=True, metadata={"help": "Wether to normalize the audio segments. Defaults to True."}
    )

    augment: bool = field(
        default=False,
        metadata={"help": "Add background noise and reverberation to recorded meetings. Defaults to False."},
    )

    bn_path: str = field(default=None, metadata={"help": "path to background noise samples. Default to None"})

    ir_path: str = field(default=None, metadata={"help": "path to impulse response samples. Default to None"})

    overlap_proba: float = field(
        default=0.3,
        metadata={"help": "Probability of adding overlap to successive audio segments. Defaults to 0.3."},
    )

    overlap_length: float = field(
        default=3,
        metadata={"help": "Maximum overlap time (in seconds) between two overlapping audio segments. Defaults to 3."},
    )

    random_gain: bool = field(
        default=False, metadata={"help": "Apply random gain to each audio segments. Defaults to False."}
    )

    add_silence: bool = field(
        default=False, metadata={"help": "Add silence or not in generated meeting . Defaults to True."}
    )

    silence_duration: int = field(
        default=3, metadata={"help": "Maximum silence duration (in seconds). Defaults to 3."}
    )

    silence_proba: int = field(
        default=0.7, metadata={"help": "Probability of adding a silence in a generated meeting. Defaults to 0.7."}
    )

    denoise: bool = field(
        default=False, metadata={"help": "Whether to denoise or not the generated meeting. Defaults to False."}
    )


@dataclass
class AdditionalArguments:

    num_proc: int = field(default=2, metadata={"help": "Number of processes used by the pipeline. Defaults to 2."})

    push_to_hub: bool = field(
        default=True, metadata={"help": "Wether to push the synthetic dataset to the hub or not. Defualt to True."}
    )

    hub_repository: str = field(
        default=None, metadata={"help": "Name of the hub repository to which the synthetic dataset will be pushed."}
    )


if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")

    parser = HfArgumentParser((ASRDatasetArguments, SyntheticMeetingArguments, AdditionalArguments))

    asr_dataset_params, synthetic_dataset_params, addition_params = parser.parse_args_into_dataclasses()

    synthetic_config = SyntheticDatasetConfig(
        dataset_name=asr_dataset_params.dataset_name,
        subset=asr_dataset_params.subset,
        split=asr_dataset_params.split,
        speaker_column_name=asr_dataset_params.speaker_column_name,
        audio_column_name=asr_dataset_params.audio_column_name,
        min_samples_per_speaker=asr_dataset_params.min_samples_per_speaker,
        nb_speakers_from_dataset=asr_dataset_params.nb_speakers_from_dataset,
        sample_rate=asr_dataset_params.sample_rate,
        num_meetings=synthetic_dataset_params.num_meetings,
        nb_speakers_per_meeting=synthetic_dataset_params.nb_speakers_per_meeting,
        segments_per_meeting=synthetic_dataset_params.segments_per_meeting,
        normalize=synthetic_dataset_params.normalize,
        augment=synthetic_dataset_params.augment,
        overlap_proba=synthetic_dataset_params.overlap_proba,
        overlap_length=synthetic_dataset_params.overlap_length,
        random_gain=synthetic_dataset_params.random_gain,
        add_silence=synthetic_dataset_params.add_silence,
        silence_duration=synthetic_dataset_params.silence_duration,
        silence_proba=synthetic_dataset_params.silence_proba,
        denoise=synthetic_dataset_params.denoise,
        bn_path=synthetic_dataset_params.bn_path,
        ir_path=synthetic_dataset_params.ir_path,
        num_proc=addition_params.num_proc,
    )

    synthetic_dataset = SyntheticDataset(synthetic_config).generate()

    if addition_params.push_to_hub and addition_params.hub_repository is not None:
        synthetic_dataset.push_to_hub(addition_params.hub_repository)
