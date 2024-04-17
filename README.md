# 🤗 Diarizers

🤗 Diarizers is a library for fine-tuning [`pyannote`](https://github.com/pyannote/pyannote-audio/tree/main) speaker 
diarization models using the Hugging Face ecosystem. It can be used to improve performance on both English and multilingual 
diarization datasets with simple example scripts, with as little as ten hours of labelled diarization data and just 5 minutes
of GPU compute time.

## 📖 Quick Index
* [Installation](#installation)
* [Train](#train)
* [Evaluation](#evaluation)
* [Inference](#inference-with-pyannote)
* [Results](#Results)
* [Adding new datasets](#adding-new-datasets)
* [Acknowledgements](#acknowledgements)
* [Citation](#citation)

## Installation

First, clone the repository and install the dependencies:

```sh
git clone https://github.com/huggingface/diarizers.git
cd diarizers
pip install -e .
```

To load pre-trained diarization models from the Hub, you'll first need to accept the terms-of-use for the following two models:
1. [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
2. [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)

And subsequently use a Hugging Face authentication token to log in with: 

```
huggingface-cli login
```

## Train

The script [`train_segmentation.py`](train_segmentation.py) can be used to pre-process a diarization dataset and subsequently
fine-tune the `pyannote` segmentation model. In the following example, we fine-tune the segmentation model on the Japanese
subset of the [CallHome dataset](https://huggingface.co/datasets/diarizers-community/callhome), a conversational dataset
between native speakers:

```bash
python3 train_segmentation.py \
    --dataset_name=diarizers-community/callhome \
    --dataset_config_name=jpn \
    --split_on_subset=data \
    --model_name_or_path=pyannote/segmentation-3.0 \
    --output_dir=./speaker-segmentation-fine-tuned-callhome-jpn \
    --do_train \
    --do_eval \
    --learning_rate=1e-3 \
    --num_train_epochs=5 \
    --lr_scheduler_type=cosine \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --preprocessing_num_workers=2 \
    --dataloader_num_workers=2 \
    --logging_steps=100 \
    --load_best_model_at_end \
    --push_to_hub
```

On a single NVIDIA RTX 24GB GPU, training takes approximately 5 minutes and improves the diarization error rate (DER) 
from 27% to 19%, representing a 29% relative improvement in performance. The final model will be pushed to the Hugging Face
Hub, for example the checkpoint [diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn](https://huggingface.co/diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn).

We encourage you to swap the CallHome Japanese dataset for a dataset in your language of choice. The [CallHome dataset](https://huggingface.co/datasets/diarizers-community/callhome)
provides splits for four additional languages, and there are a number of [other diarization datasets](https://huggingface.co/datasets?search=diarizers-community) 
available on the Hugging Face Hub. We also provide [instructions](#adding-new-datasets) for adding new datasets. 

To train on a different dataset, simply change the arguments:

- `dataset_name`: Specify a dataset from the Hub on which to fine-tune your model.  
- `dataset_config_name`: If the dataset contains multiple language subsets, select the language ID of the subset you want to train on.

If the data set doesn't contain a train and a validation split, you can automatically split it into train-val-test 
(90-10-10) by setting the argument: 

- `split_on_subset`: Specify the subset of the dataset you want to split into train-val-set.

> [!IMPORTANT]
> For now, this library can only be used to fine-tune the [segmentation model](https://huggingface.co/pyannote/segmentation-3.0) from the [speaker diarization pipeline](https://huggingface.co/pyannote/speaker-diarization-3.1). 
> Future work will aim to help optimise the hyperparameters of the entire pipeline. 

## Evaluation

The script [`test_segmentation.py`](test_segmentation.py) can be used to evaluate a fine-tuned model on a diarization
dataset. In the following example, we evaluate the fine-tuned model from the previous step on the test split of the 
CallHome Japanese dataset:

```bash
python3 test_segmentation.py \
    --dataset_name=diarizers-community/callhome \
    --dataset_config_name=jpn \ 
    --split_on_subset=data \
    --test_split_name=test \
    --model_name_or_path=diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn \
    --preprocessing_num_workers=2
```

## Inference with pyannote

The fine-tuned segmentation model can easily be loaded into the `pyannote` speaker diarization pipeline for inference. 
To do so, we load the pre-trained segmentation pipeline, and subsequently override the segmentation model with our 
fine-tuned checkpoint:

```python
from diarizers import SegmentationModel
from pyannote.audio import Pipeline
from datasets import load_dataset
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# load the pre-trained pyannote pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(device)

# replace the segmentation model with your fine-tuned one
model = SegmentationModel().from_pretrained("diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn")
model = model.to_pyannote_model()
pipeline.segmentation_model = model.to(device)

# load dataset example
dataset = load_dataset("diarizers-community/callhome", "jpn", split="data")
sample = dataset[0]["audio"]

# pre-process inputs
sample["waveform"] = torch.from_numpy(sample.pop("array")[None, :]).to(device, dtype=model.dtype)
sample["sample_rate"] = sample.pop("sampling_rate")

# perform inference
diarization_output = pipeline(sample)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
```

To apply the diarization pipeline directly to an audio file, simply call:

```python
diarization_output = pipeline("audio.wav")
```

## Adding new datasets

In order to be compatible with our training script, the Hugging Face dataset should contain the following features: 

- `audio`: Audio feature.
- `speakers`: The list of audio speakers, with their order of appearance.
- `timestamps_start`: A list of timestamps indicating the start of each speaker segment.
- `timestamps_end`: A list of timestamps indicating the end of each speaker segment.

We added several speaker-diarization datasets to the hub in the [diarizers-community](https://huggingface.co/diarizers-community) organisation. 
These datasets have been generated using the scripts in `datasets/spd_datasets.py` : The idea is to convert any raw speaker diarization dataset containing <audio, annotation> pairs into a Hugging Face dataset. 

See [Adding a dataset](datasets/README.md) for more details on how to add speaker diarization datasets to the hub. 

## Acknowledgements

This library builds on top of `pyannote` library as well as several Hugging Face libraries ([`transformers`](https://github.com/huggingface/transformers), [`datasets`](https://github.com/huggingface/datasets), [`accelerate`](https://github.com/huggingface/accelerate)). 
We would like to extend our warmest thanks to their developers!

## Citation

If you found this repository useful, please consider citing this work and also the original `pyannote` paper:

```
@misc{akesbi-diarizers,
  author = {Kamil Akesbi and Sanchit Gandhi},
  title = {"Diarizers: A repository for fine-tuning speaker diarization models"},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diarizers}}
}
```

```
@inproceedings{bredin-pyannote,
  author={Hervé Bredin},
  title={"pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe"},
  year={2023},
  booktitle={Proc. INTERSPEECH 2023},
}
```