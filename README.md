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
   --dataset_name="kamilakesbi/real_ami_ihm" \
   --pretrained_or_finetuned='finetuned' \
   --checkpoint_path='checkpoints/ami'
```


## Use in a Pyannote pipeline: 

- Use the fine-tuned segmentation model for inference: 

```python
from diarizers.models.segmentation.hf_model import SegmentationModel
from pyannote.audio import Inference

input_audio = {'waveform': waveform, 'sample_rate': sample_rate}

model = SegmentationModel().from_pretrained('checkpoints/ami')
model = model.to_pyannote_model()

# Inference result: 
segmentation_output = Inference(model, step=2.5)(input_audio)
```

- Use the fine-tuned segmentation model in a speaker diarization pipeline: 


```python

from diarizers.models.segmentation.hf_model import SegmentationModel
from pyannote.audio import Pipeline

input_audio = {'waveform': waveform, 'sample_rate': sample_rate}

pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )

# replace the segmentation model with yours: 
model = SegmentationModel().from_pretrained('checkpoints/ami')
model = model.to_pyannote_model()
pipeline.segmentation_model = model


diarization_output = pipeline(input_audio)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
```


