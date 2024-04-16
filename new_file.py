import argparse
import os
from typing import Dict, List, Optional, Union
from pyannote.audio import Model
from transformers import Trainer, TrainingArguments, HfArgumentParser

from datasets import load_dataset
from src.diarizers.data import Preprocess
from src.diarizers.models.segmentation import SegmentationModel
from src.diarizers.utils import DataCollator, Metrics, train_val_test_split
from dataclasses import dataclass, field

# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """

#     model_name_or_path: str = field(
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
#     )


# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.

#     Using `HfArgumentParser` we can turn this class
#     into argparse arguments to be able to specify them on
#     the command line.
#     """

#     dataset_name: str = field(
#         default='kamilakesbi.callhome'
#         metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     )
#     dataset_config_name: str = field(
#         default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     )
#     train_split_name: str = field(
#         default="train+validation",
#         metadata={
#             "help": (
#                 "The name of the training data set split to use (via the datasets library). Defaults to "
#                 "'train+validation'"
#             )
#         },
#     )


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))


    parser = argparse.ArgumentParser()
    # dataset arguments:
    parser.add_argument("--dataset_name", help="", default="kamilakesbi/cv_for_spd_ja_2k_rayleigh")
    # Preprocess arguments:
    parser.add_argument("--already_processed", help="", default=False)

    # Model Arguments:
    parser.add_argument("--from_pretrained", help="", default=True)

    # Training Arguments:
    parser.add_argument("--lr", help="", default=1e-3)
    parser.add_argument("--batch_size", help="", default=32)
    parser.add_argument("--epochs", help="", default=3)

    # Test arguments:
    parser.add_argument("--do_init_eval", help="", default=True)
    parser.add_argument("--checkpoint_path", help="", default="checkpoints/cv_for_spd_ja_2k_rayleigh")
    parser.add_argument("--save_model", help="", default=True)

    # Train-Test split:
    parser.add_argument("--do_split", default=False)

    # Hardware args:
    parser.add_argument("--num_proc", help="", default=24)

    args = parser.parse_args()

    dataset = load_dataset(str(args.dataset_name), num_proc=int(args.num_proc))

    if args.do_split is True:
        dataset = train_val_test_split(dataset["data"])

    model = SegmentationModel()

    if args.from_pretrained is True:
        pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
        model.from_pyannote_model(pretrained)

    if args.already_processed is True:
        preprocessed_dataset = dataset
    else:
        preprocessed_dataset = Preprocess(dataset, model).preprocess_dataset(num_proc=int(args.num_proc))

    train_dataset = preprocessed_dataset["train"].with_format("torch")
    eval_dataset = preprocessed_dataset["validation"].with_format("torch")

    metrics = Metrics(model.specifications)

    training_args = TrainingArguments(
        output_dir=str(args.checkpoint_path),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        dataloader_num_workers=int(args.num_proc),
        num_train_epochs=int(args.epochs),
        logging_steps=200,
        load_best_model_at_end=True,
        push_to_hub=False,
        save_safetensors=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(max_speakers_per_chunk=3),
        eval_dataset=eval_dataset,
        compute_metrics=metrics.der_metric,
    )

    if args.do_init_eval is True:
        first_eval = trainer.evaluate()
        print("Initial metric values: ", first_eval)
    trainer.train()

    if args.save_model is True:
        trainer.save_model(output_dir=str(args.checkpoint_path))
