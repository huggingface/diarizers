---
license: mit
base_model: pyannote/segmentation-3.0
tags:
- generated_from_trainer
datasets:
- diarizers-community/callhome
model-index:
- name: speaker-segmentation-fine-tuned-callhome-jpn
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# speaker-segmentation-fine-tuned-callhome-jpn

This model is a fine-tuned version of [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) on the diarizers-community/callhome jpn dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6122
- Der: 0.2052
- False Alarm: 0.0800
- Missed Detection: 0.0739
- Confusion: 0.0512

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Der    | False Alarm | Missed Detection | Confusion |
|:-------------:|:-----:|:----:|:---------------:|:------:|:-----------:|:----------------:|:---------:|
| 0.6365        | 1.0   | 336  | 0.6122          | 0.2052 | 0.0800      | 0.0739           | 0.0512    |


### Framework versions

- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.2
