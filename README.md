# Diarizers


## Datasets:

Features: 


- `audio`:
- `timestamps_start`:
- `timestamps_end`:
- `speakers`:


### Add a speaker diarization dataset to the Hub: 




### Preprocess the dataset:

Preprocess the dataset to use it with the hugging face `Trainer`: 

```python 

from datasets import load_dataset
from diarizers.models.segmentation import SegmentationModelConfig, SegmentationModel
from diarizers.data.preprocess import Preprocess

ds = load_dataset("kamilakesbi/callhome", "jpn", num_proc=8)

config = SegmentationModelConfig(
    chunk_duration=10, 
    max_speakers_per_frame=2, 
    max_speakers_per_chunk=3, 
    min_duration=None, 
    warm_up=(0.0, 0.0),
    weigh_by_cardinality=False
)

model = SegmentationModel(config=config)

preprocessed_dataset = Preprocess(ds, model).preprocess_dataset(num_proc=8)
preprocessed_dataset.push_to_hub("your_repository")
```


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


## Use in pyannote: 

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


