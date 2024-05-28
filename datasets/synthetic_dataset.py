import torch
from dataclasses import dataclass, field
from diarizers import SyntheticDataset, SyntheticDatasetConfig
from transformers import HfArgumentParser


@dataclass
class ASRDatasetArguments:
    """
    """

    dataset_name: str = field(
        default="mozilla-foundation/common_voice_17_0",
        metadata={"help": "ASR dataset with single speaker audio files to be used to generate synthetic speaker diarization meetings. Defaults to 'mozilla-foundation/common_voice_17_0'."}
    )
    subset: str = field(
        default='validated', metadata={"help": "ASR dataset subset. Defaults to 'validated'."}
    )

    split: str = field(
        default="ja", metadata={"help": "ASR dataset split. Defaults to 'ja'"}
    )

    speaker_column_name: str = field(
        default="client_id", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    audio_column_name: str = field(
        default="audio", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    min_samples_per_speaker: int = field(
        default=10, metadata={"help": "Minimum number of audio samples from a given speaker in the ASR dataset to use that speaker for meeting generation. Defaults to 10"}
    )

    nb_speakers_from_dataset: int = field(
        default=200, metadata={"help": "Maximum number of speakers to use from the ASR dataset to generate meetings"}
    )

    sample_rate: int = field(
        default=16000, metadata={"help": "Generated meetings sample rate"}
    )



@dataclass
class SyntheticMeetingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    nb_speakers_per_meeting: int = field(
        default=3,
        metadata={"help": "number of speakers in generated meeting. Defaults to 3."}
    )
    num_meetings: int = field(
        default=200, metadata={"help": "Number of meeting audio files to generate. Defaults to 1000."}
    )

    segments_per_meeting: int = field(
        default=16, metadata={"help": "number of audio segments used in a generated meeting. Defaults to 16."}
    )

    normalize: bool = field(
        default=True, metadata={"help": "normalize audio segments. Defaults to True."}
    )


    augment: bool = field(
        default=False, metadata={"help": "augment generated meetings with background noise and reverberation. Defaults to False."}
    )

    overlap_proba: float = field(
        default=0.3, metadata={"help": "Probability of adding overlap to concatenated consecutive audio segments. Defaults to 0.3."}
    )

    overlap_length: float = field(
        default=3, metadata={"help": "Maximum overalp duration (in seconds) between two overlapping audio segments. Defaults to 3."}
    )

    random_gain: bool = field(
        default=False, metadata={"help": "Apply random gain to each audio segments. Defaults to False."}
    )

    add_silence: bool = field(
        default=False, metadata={"help": "Add silence or not in generated meeting . Defaults to True."}
    )

    silent_duration: int = field(
        default=3, metadata={"help": "maximum silence duration (in seconds). Defaults to 3."}
    )

    silent_proba: int = field(
        default=3, metadata={"help": "probability of adding a silence in a generated meeting. Defaults to 3."}
    )

    denoise: bool = field(
        default=False, metadata={"help": "Denoise the generated meeting. Defaults to False."}
    )

    bn_path: str = field(
        default=None, metadata={"help": "path to background noise samples."}
    )

    bn_path: str = field(
        default=None, metadata={"help": "path to impulse response samples"}
    )
    


@dataclass
class AdditionalArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    num_proc: int = field(
        default=2, metadata={"help": "Number of processors used by the pipeline. Defaults to 2"}
    )


if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")

    parser = HfArgumentParser((ASRDatasetArguments, SyntheticMeetingArguments, AdditionalArguments))
        
    synthetic_config = SyntheticDatasetConfig(**parser)
    synthetic_dataset = SyntheticDataset(synthetic_config)

