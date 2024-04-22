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

