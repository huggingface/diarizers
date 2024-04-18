import os

from pyannote.audio import Model
from datasets import load_dataset
from diarizers import SegmentationModel, Test, train_val_test_split
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

@dataclass
class DataTrainingArguments:
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
    dataset_config_name: str = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    test_split_name: str = field(
        default="test", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    split_on_subset: str = field(
        default=None,
        metadata={"help": "Automatically splits the dataset into train-val-set on a specified subset. Defaults to 'None'"},
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = HfArgumentParser((DataTrainingArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()

    # Load the dataset: 
    if str(data_args.dataset_config_name): 
        dataset = load_dataset(
            str(data_args.dataset_name), 
            str(data_args.dataset_config_name), 
            num_proc=int(data_args.preprocessing_num_workers)
        )
    else: 
        dataset = load_dataset(
            str(data_args.dataset_name), 
            str(data_args.dataset_config_name), 
            num_proc=int(data_args.preprocessing_num_workers)
    )
        
    test_split_name = data_args.test_split_name
    if data_args.split_on_subset:
        dataset = train_val_test_split(dataset[str(data_args.split_on_subset)])
        test_split_name = 'test'

    test_dataset = dataset[data_args.test_split_name]

    if model_args.model_name_or_path == "pyannote/segmentation-3.0": 
        model = Model.from_pretrained(model_args.model_name_or_path, use_auth_token=True)
    else: 
        model = SegmentationModel()
        model = model.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,  
            use_auth_token=True
        )
        model = model.to_pyannote_model()

    test = Test(test_dataset, model, step=2.5)
    metrics = test.compute_metrics()
    print(metrics)