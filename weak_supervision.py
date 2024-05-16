import os 
from datasets import load_dataset
from pyannote.audio import Model
import torch 
import numpy as np 

from pyannote.audio.torchmetrics import (DiarizationErrorRate, FalseAlarmRate,
                                         MissedDetectionRate,
                                         SpeakerConfusionRate)
from pyannote.audio.utils.powerset import Powerset

class WeakSupervision():

    def __init__(self) -> None:

        self.max_speakers_per_frame = 2
        self.max_speakers_per_chunk = 3

        self.powerset = Powerset(
            self.max_speakers_per_chunk, 
            self.max_speakers_per_frame, 
        )

        self.metrics = {
            "der" : DiarizationErrorRate(0.5), 
            'false_alarm' : FalseAlarmRate(0.5), 
            'missed_detection' : MissedDetectionRate(0.5), 
            'confusion' : SpeakerConfusionRate(0.5), 
        }

    def filter_rows(self, file): 

        new_batch = {
                "der": [],
                "fa_metric": [],
                "md_metric": [],
                "sc_metric": [],
        }

        waveform = torch.tensor(file['waveforms']).unsqueeze(0).unsqueeze(0)
        labels = torch.tensor(file['labels'])
        predictions = model(waveform)

        predictions = self.powerset.to_multilabel(predictions)

        labels = torch.tensor(labels)
        predictions = torch.transpose(predictions, 1, 2)

        labels = np.pad(
                labels,
                ((0, 0), (0, 3 - labels.shape[1])),
                mode="constant",
            )

        labels = torch.from_numpy(np.stack(labels)).unsqueeze(0)
        labels = torch.transpose(labels, 1, 2)

        der_metric = DiarizationErrorRate(0.5)
        fa_metric = FalseAlarmRate(0.5)
        md_metric = MissedDetectionRate(0.5)
        sc_metric = SpeakerConfusionRate(0.5)

        new_batch['der'] = float(der_metric(predictions, labels))
        new_batch['false_alarm'] = float(fa_metric(predictions, labels))
        new_batch['missed_detection'] = float(md_metric(predictions, labels))
        new_batch['confusion'] = float(sc_metric(predictions, labels))

        return new_batch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

processed = load_dataset('kamilakesbi/processed_test')['train']
model = Model.from_pretrained('pyannote/segmentation-3.0')

weak_supervision = WeakSupervision()

processed = processed.select(range(1,100)).map(weak_supervision.filter_rows, num_proc=12)
processed = processed.filter(lambda x:x['false_alarm']<=0.05)
print(processed)