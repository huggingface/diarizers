# Diarizers


## Datasets:

Features: 


- `audio`:
- `timestamps_start`:
- `timestamps_end`:
- `speakers`:


### Add a speaker Diarization dataset to the Hub: 



### Synthetic datasets: 



## Fine-Tune: 

```
python3 train_segmentation.py
    --dataset_name="kamilakesbi/real_ami_ihm" \
    --from_pretrained=True \
    --lr='1e-3'\
    --batch_size='32'\
    --epochs='3'\
    --do_init_eval=True\
    --checkpoint_path='checkpoints/ami'\
    --save_model=True \
    --num_proc='12'
```

## Test: 

```
python3 test_segmentation.py \
   --dataset_name="kamilakesbi/real_ami_ihm"\
   --pretrained_or_finetuned='finetuned'\
   --checkpoint_path='checkpoints/ami'\
```


## Use in a Pyannote pipeline: 




