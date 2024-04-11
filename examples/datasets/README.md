# Speaker Diarization Datasets:  

To reproduce the datasets present in the [Speaker Diarization Collection](https://huggingface.co/collections/kamilakesbi/speaker-diarization-datasets-660d2b4fff9745457c89e164), we used the following scripts: 

## AMI IHM AND SDM: 

```
git clone https://github.com/pyannote/AMI-diarization-setup.git
cd /AMI-diarization-setup/pyannote/
sh download_ami.sh
sh download_ami_sdm.sh
```

Use this script to compute and push the AMI dataset to the hub: 

```
python3 -m examples.datasets.ami \
    --path_to_ami=/path_to_ami \
    --push_to_hub=True \
    --hub_repository=kamilakesbi/ami
```

## CALLHOME: 

Download for each langage (example here japanese): 

```
!wget https://ca.talkbank.org/data/CallHome/jpn.zip
!wget -r -np -nH --cut-dirs=2 -R index.html* https://media.talkbank.org/ca/CallHome/jpn/
!unzip jpn.zip
```

Use this script to compute and push the `Japanese, English, Chinese, German and Spanish Callhome` dataset to the hub: 

```
python3 -m examples.datasets.callhome \
    --path_to_callhome=/home/kamil/datasets \
    --push_to_hub=True \
    --hub_repository=kamilakesbi/callhome \
```

## VOXCONVERSE: 

```
git clone git@github.com:joonson/voxconverse.git
https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip
```


```
python3 -m examples.datasets.voxconverse 
    --path_to_voxconverse=/path_to_voxconverse
    --push_to_hub=True 
    --hub_repository=speaker_diarization/voxconverse
```

## SIMSAMU: 


