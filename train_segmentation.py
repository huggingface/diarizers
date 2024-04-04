import argparse
import os

from datasets import load_dataset
from diarizers.models.segmentation.hf_model import SegmentationModel
from diarizers.data.preprocess import Preprocess
from transformers import Trainer, TrainingArguments

from diarizers.utils import DataCollator, Metrics
from pyannote.audio import Model


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    parser = argparse.ArgumentParser()
    # dataset arguments:
    parser.add_argument("--dataset_name", help="", default="kamilakesbi/cv_for_spd_fr_2k_std_0.2")
    # Preprocess arguments:
    parser.add_argument("--already_processed", help="", default=False)

    # Model Arguments:
    parser.add_argument("--from_pretrained", help="", default=True)

    # Training Arguments:
    parser.add_argument("--lr", help="", default=1e-3)
    parser.add_argument("--batch_size", help="", default=32)
    parser.add_argument("--epochs", help="", default=1)

    # Test arguments:
    parser.add_argument("--do_init_eval", help="", default=True)
    parser.add_argument('--checkpoint_path', help="", default='checkpoints/cv_for_spd_fr_2k_std_0.2')
    parser.add_argument('--save_model', help="", default=True)

    # Hardware args: 
    parser.add_argument('--num_proc', help="", default=24)

    args = parser.parse_args()

    dataset = load_dataset(str(args.dataset_name), num_proc=int(args.num_proc))
    
    model = SegmentationModel()

    if args.from_pretrained is True:
        pretrained = Model.from_pretrained(
            "pyannote/segmentation-3.0", use_auth_token=True
        )
        model.from_pyannote_model(pretrained)
    
    if args.already_processed is True: 
        preprocessed_dataset = dataset
    else: 
        preprocessed_dataset = Preprocess(
            dataset, model
        ).preprocess_dataset(num_proc=int(args.num_proc))
    
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
