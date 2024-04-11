# Dataset download before preprocesssing: 

How to reproduce the datasets present in the [Speaker Diarization Collection](https://huggingface.co/collections/kamilakesbi/speaker-diarization-datasets-660d2b4fff9745457c89e164): 

## AMI IHM AND SDM: 

```
git clone https://github.com/pyannote/AMI-diarization-setup.git
cd /AMI-diarization-setup/pyannote/
sh download_ami.sh
```

Use this script to compute and push the AMI dataset to the hub: 

```
python3 -m examples.datasets.ami_dataset 
    --path_to_ami=/path_to_ami 
    --push_to_hub=True 
    --hub_repository=speaker_diarization/ami
```


## CALLHOME: 

Japanese: 

```
!wget https://ca.talkbank.org/data/CallHome/jpn.zip
!wget -r -np -nH --cut-dirs=2 -R index.html* https://media.talkbank.org/ca/CallHome/jpn/
!unzip jpn.zip
```

Push the dataset to hub: 

```
python3 callhome_dataset.py --langage=jpn
```

Same steps to download and push to hub the spanish, english, german or chinese versions of the callhome dataset. 

## SIMSAMU: 



## VOXCONVERSE: 