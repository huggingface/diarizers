
from datasets import load_dataset
from src.diarizers import SegmentationModelConfig, SegmentationModel, Preprocess

ds = load_dataset("kamilakesbi/ami", "ihm", num_proc=24)

config = SegmentationModelConfig(
    chunk_duration=10,
    max_speakers_per_frame=2,
    max_speakers_per_chunk=3,
    min_duration=None,
    warm_up=(0.0, 0.0),
    weigh_by_cardinality=False
)

model = SegmentationModel(config=config)

preprocessor = Preprocess(config)

preprocessed_dataset = ds['validation'].map(
    lambda file: preprocessor(file, random=False, overlap=0.0), 
    num_proc=1, 
    remove_columns=next(iter(ds.values())).column_names,
    keep_in_memory=True, 
    batched=True, 
    batch_size=1
)

print(preprocessed_dataset)