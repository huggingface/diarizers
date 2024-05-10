import os

from pyannote.audio import Model, Pipeline
from datasets import load_dataset, DatasetDict
from diarizers import SegmentationModel, Test, TestPipeline
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

@dataclass
class EvaluateArguments:
    """
    Arguments to .
    """

    evaluate_with_pipeline: bool = field(
        default=False, 
        metadata={"help": "Compute metrics using the full speaker diarization pipeline with modified speaker segmentation model"}
    )


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, EvaluateArguments))
    data_args, model_args, evaluate_args = parser.parse_args_into_dataclasses()

    # Load the Dataset:
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

    # Split in Train-Val-Test and use Test Subset:
    if data_args.split_on_subset:
        
        train_testvalid = dataset[str(data_args.split_on_subset)].train_test_split(test_size=0.2, seed=0)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=0)

        dataset = DatasetDict({
            'train': train_testvalid['train'],
            'validation': test_valid['test'],
            'test': test_valid['train']}
        )
        test_split_name = 'test'

    test_dataset = dataset[data_args.test_split_name]

    # Load the Pretrained or Fine-Tuned segmentation model:
    if model_args.model_name_or_path == "pyannote/segmentation-3.0": 
        model = Model.from_pretrained(model_args.model_name_or_path, use_auth_token=True)
    else:
        model = SegmentationModel().from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True
        )
        model = model.to_pyannote_model()

    # Test and Print Metrics:
    print('Segmentation Model evaluation: ')
    test = Test(test_dataset, model, step=2.5)
    metrics = test.compute_metrics()
    print(metrics)

    # Pipeline:
    if evaluate_args.evaluate_with_pipeline:
        print('Speaker diarization pipeline (with fine-tuned segmentation model) evaluation:')
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        pipeline._segmentation.model = model

        pipeline_metrics = TestPipeline(test_dataset, pipeline).compute_metrics()
        print(pipeline_metrics)