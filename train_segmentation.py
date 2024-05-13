import os
from typing import Optional
from pyannote.audio import Model
from transformers import Trainer, TrainingArguments, HfArgumentParser

from datasets import load_dataset, DatasetDict
from diarizers import Preprocess, SegmentationModel,  DataCollator, Metrics
from dataclasses import dataclass, field

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

    train_split_name: str = field(
        default="train", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"}
    )

    eval_split_name: str = field(
        default="validation", metadata={"help": "The name of the training data set split to use (via the datasets library). Defaults to 'val'"}
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

    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, TrainingArguments))

    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Load the Dataset:
    if data_args.dataset_config_name: 
        dataset = load_dataset(
            str(data_args.dataset_name), 
            str(data_args.dataset_config_name), 
            num_proc=int(data_args.preprocessing_num_workers)
        )
    else: 
        dataset = load_dataset(
            str(data_args.dataset_name), 
            num_proc=int(data_args.preprocessing_num_workers)
        )
   
    train_split_name = data_args.train_split_name
    val_split_name = data_args.eval_split_name

    # Split in Train-Val-Test:
    if data_args.split_on_subset:
        
        train_testvalid = dataset[str(data_args.split_on_subset)].train_test_split(test_size=0.2, seed=0)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=0)

        dataset = DatasetDict({
            'train': train_testvalid['train'],
            'validation': test_valid['test'],
            'test': test_valid['train']}
        )
        train_split_name = 'train'
        val_split_name = 'validation'

    # Load the Pretrained Pyannote Segmentation Model:
    pretrained = Model.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,  
        use_auth_token=True
    )
    model = SegmentationModel.from_pyannote_model(pretrained)

    # Load the preprocessor: 
    preprocessor = Preprocess(model.config)

    # Preprocess:
    if training_args.do_train:
        train_set = dataset[str(train_split_name)].map(
            lambda file: preprocessor(file, random=False, overlap=0.5), 
            num_proc=data_args.preprocessing_num_workers, 
            remove_columns=next(iter(dataset.values())).column_names,
            batched=True, 
            batch_size=1
        ).with_format("torch")

    if training_args.do_eval: 
        val_set = dataset[str(val_split_name)].map(
            lambda file: preprocessor(file, random=False, overlap=0.0), 
            num_proc=data_args.preprocessing_num_workers, 
            remove_columns=next(iter(dataset.values())).column_names,
            batched=True, 
            keep_in_memory=True, 
            batch_size=1
        ).with_format('torch')

    # Load metrics:
    metrics = Metrics(model.specifications)

    # Define the Trainer:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        data_collator=DataCollator(max_speakers_per_chunk=model.config.max_speakers_per_chunk),
        eval_dataset=val_set,
        compute_metrics=metrics,
    )

    # Train!
    if training_args.do_eval:
        first_eval = trainer.evaluate()
        print("Initial metric values: ", first_eval)
    if training_args.do_train:
        trainer.train()

    # Write Training Stats
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "speaker diarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
    kwargs['tags'] = ['speaker-diarization', 'speaker-segmentation']

    # Push to Hub
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
