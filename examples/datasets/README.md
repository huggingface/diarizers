# Dataset download before preprocesssing: 


## AMI IHM: 



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