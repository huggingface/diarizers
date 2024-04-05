from datasets import load_dataset
from diarizers.models.segmentation.hf_model import SegmentationModel
from pyannote.audio import Model
from diarizers.test import Test


dataset = load_dataset('kamilakesbi/callhome_jpn')

example = dataset['data'].select(range(1))

model = SegmentationModel()
model = model.from_pretrained("checkpoints/cv_for_spd_ja_2k_rayleigh")
model = model.to_pyannote_model()

test = Test(example, model, step=2.5)
metrics = test.compute_metrics()
print(metrics)

model = Model.from_pretrained(
        "pyannote/segmentation-3.0", use_auth_token=True
    )

test = Test(example, model, step=2.5)
metrics = test.compute_metrics()
print(metrics)
