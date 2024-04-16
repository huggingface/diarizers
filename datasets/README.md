### Add a speaker diarization dataset to the Hub: 

To reproduce the datasets present in the [Speaker Diarization Collection](https://huggingface.co/collections/kamilakesbi/speaker-diarization-datasets-660d2b4fff9745457c89e164), we used the following scripts: 


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

```
python3 -m spd_datasets \
    --dataset=callhome \
    --path_to_callhome=/home/kamil/datasets \
    --push_to_hub=True \
    --hub_repository=kamilakesbi/callhome \
```

