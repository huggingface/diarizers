import torch

from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    SpeakerConfusionRate,
)
from pyannote.audio.utils.powerset import Powerset


class Metrics:
    def __init__(self, specifications) -> None:

        self.powerset = specifications.powerset
        self.classes = specifications.classes
        self.powerset_max_classes = specifications.powerset_max_classes

        self.model_powerset = Powerset(
            len(self.classes),
            self.powerset_max_classes,
        )

        self.metrics = {
            "der": DiarizationErrorRate(0.5),
            "confusion": SpeakerConfusionRate(0.5),
            "missed_detection": MissedDetectionRate(0.5),
            "false_alarm": FalseAlarmRate(0.5),
        }

    def der_metric(self, eval_pred):

        logits, labels = eval_pred

        if self.powerset:
            predictions = self.model_powerset.to_multilabel(torch.tensor(logits))
        else:
            predictions = torch.tensor(logits)

        labels = torch.tensor(labels)

        predictions = torch.transpose(predictions, 1, 2)
        labels = torch.transpose(labels, 1, 2)

        metrics = {"der": 0, "false_alarm": 0, "missed_detection": 0, "confusion": 0}

        metrics["der"] += self.metrics["der"](predictions, labels).cpu().numpy()
        metrics["false_alarm"] += (
            self.metrics["false_alarm"](predictions, labels).cpu().numpy()
        )
        metrics["missed_detection"] += (
            self.metrics["missed_detection"](predictions, labels).cpu().numpy()
        )
        metrics["confusion"] += (
            self.metrics["confusion"](predictions, labels).cpu().numpy()
        )

        return metrics

