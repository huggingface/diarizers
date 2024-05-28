import os
import torch
from dataclasses import dataclass, field
from synthetic.synthetic import SyntheticDataset
from transformers import HfArgumentParser


def pickle_model(model):
    if not os.path.exists("vad.pt"):
        model.save("vad.pt")
    return torch.jit.load, ("vad.pt",)


@dataclass
class ASRDatasetArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    subset: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    split: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    min_samples_per_speaker: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    speaker_column_name: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    audio_column_name: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    nb_speakers_from_dataset: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )


@dataclass
class SyntheticMeetingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    nb_speakers_per_meeting: str = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    num_meetings: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    segments_per_meeting: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    next_speaker_proba: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    random_volume: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    normalize: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    augment: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    bn_path: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    denoise: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    silent_regions: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    silent_duration: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    silent_proba: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    overlap_proba: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    overlap_length: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )



if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")

    parser = HfArgumentParser((ASRDatasetArguments, SyntheticMeetingArguments))



    config = {
        "dataset": {
            "dataset_name": "mozilla-foundation/common_voice_17_0",
            "split": "ja",
            "subset": "validated",
            "speaker_column_name": "client_id",
            "audio_column_name": "audio",
            "min_samples_per_speaker": 10,
            "nb_speakers_from_dataset": -1,
        },
        "meeting": {
            "nb_speakers_per_meeting": 2,
            "num_meetings": 16000,
            "segments_per_meeting": 16,
            "next_speaker_proba": 0,
            "random_volume": False,
            "normalize": True,
            "augment": False,
            "denoise": False,
            "silence": {
                "silent_regions": True,
                "silent_duration": 3,
                "silent_proba": 0.5,
            },
            "overlap": {
                "overlap_proba": 0.3,
                "overlap_length": 3,
            },
            "bn_path": "/home/kamil/datasets/wham_noise/wham_noise/tr",
            "ir_path": "/home/kamil/datasets/MIT-ir-survey",
            "sample_rate": 16000,
        },
        "num_proc": 24,
    }

    synthetic_dataset = SyntheticDataset(
        config,
    ).create_spd_dataset()

    synthetic_dataset.push_to_hub("kamilakesbi/synthetic_dataset_jpn_2_Big_10")
