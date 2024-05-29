from dataclasses import dataclass, field

import torch
from transformers import HfArgumentParser

from diarizers import SyntheticDataset, SyntheticDatasetConfig


@dataclass
class ASRDatasetArguments:
    """ """

    dataset_name: str = field(
        default="mozilla-foundation/common_voice_17_0",
        metadata={
            "help": "ASR dataset with single speaker audio files to be used to generate synthetic speaker diarization meetings. Defaults to 'mozilla-foundation/common_voice_17_0'."
        },
    )
    subset: str = field(default="validated", metadata={"help": "ASR dataset subset. Defaults to 'validated'."})

    split: str = field(default="ja", metadata={"help": "ASR dataset split. Defaults to 'ja'"})

    speaker_column_name: str = field(
        default="client_id",
        metadata={
            "help": "Speaker column name. Default to 'client_id'"
        },
    )

    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "Audio column name. Default to 'audio'"
        },
    )

    min_samples_per_speaker: int = field(
        default=10,
        metadata={
            "help": "Minimum number of audio samples from a given speaker in the ASR dataset to use that speaker for meeting generation. Defaults to 10"
        },
    )

    nb_speakers_from_dataset: int = field(
        default=200, metadata={"help": "Maximum number of speakers to use from the ASR dataset to generate meetings"}
    )

    sample_rate: int = field(default=16000, metadata={"help": "Generated meetings sample rate"})


@dataclass
class SyntheticMeetingArguments:
    """ """

    nb_speakers_per_meeting: int = field(
        default=3, metadata={"help": "number of speakers in generated meeting. Defaults to 3."}
    )
    num_meetings: int = field(
        default=200, metadata={"help": "Number of meeting audio files to generate. Defaults to 1000."}
    )

    segments_per_meeting: int = field(
        default=16, metadata={"help": "number of audio segments used in a generated meeting. Defaults to 16."}
    )

    normalize: bool = field(default=True, metadata={"help": "normalize audio segments. Defaults to True."})

    augment: bool = field(
        default=False,
        metadata={"help": "augment generated meetings with background noise and reverberation. Defaults to False."},
    )

    bn_path: str = field(default=None, metadata={"help": "path to background noise samples."})

    ir_path: str = field(default=None, metadata={"help": "path to impulse response samples"})

    overlap_proba: float = field(
        default=0.3,
        metadata={
            "help": "Probability of adding overlap to concatenated consecutive audio segments. Defaults to 0.3."
        },
    )

    overlap_length: float = field(
        default=3,
        metadata={
            "help": "Maximum overlap duration (in seconds) between two overlapping audio segments. Defaults to 3."
        },
    )

    random_gain: bool = field(
        default=False, metadata={"help": "Apply random gain to each audio segments. Defaults to False."}
    )

    add_silence: bool = field(
        default=False, metadata={"help": "Add silence or not in generated meeting . Defaults to True."}
    )

    silence_duration: int = field(
        default=3, metadata={"help": "maximum silence duration (in seconds). Defaults to 3."}
    )

    silence_proba: int = field(
        default=3, metadata={"help": "probability of adding a silence in a generated meeting. Defaults to 3."}
    )

    denoise: bool = field(default=False, metadata={"help": "Denoise the generated meeting. Defaults to False."})


@dataclass
class AdditionalArguments:
    """ """

    num_proc: int = field(default=2, metadata={"help": "Number of processors used by the pipeline. Defaults to 2"})

    push_to_hub: bool = field(default=True, metadata={"help": "push the synthetic dataset to the hub"})

    hub_repository: str = field(
        default=None, metadata={"help": "Name of the hub repository where the synthetic dataset will be pushed."}
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

    if addition_params.push_to_hub and addition_params.dataset_name is not None:
        synthetic_dataset.push_to_hub(addition_params.hub_repository)
