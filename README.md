# Diarizers

Diarizers is a library for fine-tuning [`pyannote`](https://github.com/pyannote/pyannote-audio/tree/develop) speaker diarisation models using Hugging Face tools. It provides scripts and tools to: 

- Convert Speaker Diarization datasets into Hugging Face datasets compatible with the `diarizers` library. 
- Fine-tune speaker diarization models using the Hugging Face `Trainer`. 
- Test the models at inference time. 
- Convert the fine-tuned models into a format compatible with `pyannote`.  

> [!IMPORTANT]
> For now, this library can only be used to fine-tune the [segmentation model](https://huggingface.co/pyannote/segmentation-3.0) from the [speaker diarization pipeline](https://huggingface.co/pyannote/speaker-diarization-3.1). 
> Future work will aim to help optimise the hyperparameters of the entire pipeline. 

## 📖 Quick Index
* [Installation](#installation)
* [Train](#train)
* [Inference](#inference)
* [Results](#Results)
* [Adding new datasets](#addingnewdatasets)
* [Acknowledgements](#acknowledgements)
* [Citation](#citation)

## Installation

Diarizers has light dependencies and can be installed with the following lines: 

```sh
pip install git+https://github.com/huggingface/diarizers.git
pip install diarizers[dev]
```

You'll need to generate a [user access token](https://huggingface.co/docs/hub/en/security-tokens), and login with: 

```
pip install -U "huggingface_hub[cli]"
hugging-cli login
```

## Train

When fine-tuning a `pyannote` segmentation model on a given dataset, be sure to specify this: 

- `dataset_name`: Specify a dataset from the Hub on which to fine-tune your model.  
- `dataset_config_name`:  If the dataset contains multiple language subsets, select the language ID of the subset you want to train on.
- `train_split_name`: Specify which dataset split is to be used for training. Default is 'train'. 
- `eval_split_name`: Specify which dataset split is to be used for validation. Default is 'validation'. 

If the data set doesn't already contain a train and a validation split, you can automatically split it into train-val-test (90-10-10) using: 

- `do_split_on_subset`: Specify the subset of the dataset you want to split into train-val-set.

```
python3 train_segmentation.py
    --dataset_name=diarizers-community/callhome \
    --dataset_config_name=jpn \ 
    --do_split_on_subset=data \
    --model_name_or_path=pyannote/segmentation-3.0 \
    --output_dir=diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn \
    --preprocessing_num_workers=2 \ 
    --do_train \
    --do_eval \ 
    --learning_rate=1e-3 \ 
    --num_train_epochs=1 \
    --per_device_train_batch_size=32 \ 
    --per_device_eval_batch_size=32 \ 
    --evaluation_strategy=epoch \
    --save_strategy=epoch \  
    --dataloader_num_workers=2 \ 
    --logging_steps=200 \
    --load_best_model_at_end \
    --push_to_hub
```

## Inference

Evaluate your model at inference time using: 

```
python3 test_segmentation.py \
      --dataset_name=diarizers-community/callhome" \
      --dataset_config_name=jpn" \ 
      --do_split_on_subset=data" \
      --model_name_or_path=diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn" \ 
      --preprocessing_num_workers=2  
```

## Use in pyannote

Use the fine-tuned segmentation model within a speaker diarization pipeline: 

```python
from diarizers import SegmentationModel
from pyannote.audio import Pipeline

# load the pre-trained pyannote pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# replace the segmentation model with your fine-tuned one
model = SegmentationModel().from_pretrained("diarizers-community/speaker-segmentation-fine-tuned-callhome-jpn")
model = model.to_pyannote_model()
pipeline.segmentation_model = model

# perform inference
diarization_output = pipeline("audio.mp3")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
```

## Results


## Adding new datasets

In order to be compatible with our training script, the hugging face dataset should contain the following features: 

- `audio`: Audio feature.
- `speakers`: The list of audio speakers, with their order of appearance.
- `timestamps_start`: A list of timestamps indicating the start of each speaker segment.
- `timestamps_end`: A list of timestamps indicating the end of each speaker segment.

We added several speaker-diarization datasets to the hub in the [diarizers-community](https://huggingface.co/diarizers-community) organisation. 
These datasets have been generated using the scripts in `datasets/spd_datasets.py` : The idea is to convert any raw speaker diarization dataset containing <audio, annotation> pairs into a hugging face dataset. 

See [Adding a dataset](datasets/README.md) for more details on how to add speaker diarization datasets to the hub. 

## Acknowledgements

This library builds on top of `pyannote` library as well as several hugging face libraries (`transformers`, `datasets`, `accelerate`). 
We would like to extend our warmest thanks to their developpers!


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