# Speaker diarization datasets

## Add any speaker diarization dataset to the hub

General steps to add a Speaker diarization dataset with <files, annotations> to the hub:  

1. Prepare a folder containing audios and annotations files , which should be organised like this: 

```
    dataset_folder
    ├── audio                   
    │   ├── file_1.mp3          
    │   ├── file_2.mp3          
    │   └── file_3.mp3                 
    ├── annotations          
    │   ├── file_1.rttm          
    │   ├── file_2.rttm          
    │   └── file_3.rttm    
```


2. Get dictionnaries with the following structure:

```
annotations_files = {
    "subset1": [list of annotations_files in subset1],
    "subset2":  [list of annotations_files in subset2],
}

audio_files = {
    "subset1": [list of annotations_files in subset1],
    "subset2":  [list of annotations_files in subset2],   
}
```

Here, each subset will correspond in a Hugging Face dataset subset. 

3. Use SpeakerDiarization module from `diarizers` to obtain your Hugging Face dataset: 

```
from diarizers import SpeakerDiarizationDataset

dataset = SpeakerDiarizationDataset(audio_files, annotations_files).construct_dataset()
```

Note: This module can currently be used on RTTM format annotation files, but may need to be adapted for other formats.

## Current datasets in diarizers-community

We explain the scripts we used to add the various datasets present in the [diarizers-community](https://huggingface.co/diarizers-community): 

#### AMI IHM AND SDM: 

```
git clone https://github.com/pyannote/AMI-diarization-setup.git
cd /AMI-diarization-setup/pyannote/
sh download_ami.sh
sh download_ami_sdm.sh
```

#### CALLHOME: 

Download for each language (example here for Japanese): 

```
wget https://ca.talkbank.org/data/CallHome/jpn.zip
wget -r -np -nH --cut-dirs=2 -R index.html* https://media.talkbank.org/ca/CallHome/jpn/
unzip jpn.zip
```

#### VOXCONVERSE: 

Download the RTTM files: 

```
git clone git@github.com:joonson/voxconverse.git
```

Download the audio files: 

```
wget https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
unzip voxconverse_dev_wav.zip

wget https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip
unzip voxconverse_test_wav.zip
```

#### SIMSAMU: 

The Simsamu dataset is based on this [Hugging Face dataset](https://huggingface.co/datasets/medkit/simsamu): 

```
git lfs install
git clone git@hf.co:datasets/medkit/simsamu
```

#### Push to hub: 

We pushed each of these datasets using `spd_datasets.py` and the following script: 


```
python3 spd_datasets.py \
    --dataset=callhome \
    --path_to_dataset=/path_to_callhome \
    --push_to_hub=False \
    --hub_repository=diarizers-community/callhome \
```


# Synthetic dataset generation: 



# Generate a synthetic dataset compatible with diarizers: 


## Installation

## How to use? 


"/home/kamil/datasets/wham_noise/wham_noise/tr"

"/home/kamil/datasets/MIT-ir-survey"



```bash
python3 synthetic_dataset.py \
    --dataset_name="mozilla-foundation/common_voice_17_0" \
    --subset="validated" \
    --split="ja" \
    --speaker_column_name="client_id" \
    --audio_column_name="audio" \
    --min_samples_per_speaker=10 \
    --nb_speakers_from_dataset=200 \
    --sample_rate=16000 \
    --nb_speakers_per_meeting=3 \
    --num_meetings=200 \
    --segments_per_meeting=16 \
    --normalize=True \
    --augment=False \
    --overlap_proba=0.3 \
    --overlap_length=3 \
    --random_gain=False \
    --add_silence=False \
    --silence_duration=3 \
    --silence_proba=3 \
    --denoise=False \
    --bn_path=None \
    --ir_path=None \
    --num_proc=2 \
    --push_to_hub=True \
    --hub_repository='test'
```

