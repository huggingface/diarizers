
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

preprocessor = Preprocess(ds, config)
print('ok')

preprocessed_dataset = ds.map(preprocessor, num_proc = 24)

print('ok')