import numpy as np
import torch
from pyannote.core import SlidingWindow, SlidingWindowFeature
from tqdm import tqdm

from pyannote.audio import Inference
from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.torchmetrics import (
    DiarizationErrorRate,
    FalseAlarmRate,
    MissedDetectionRate,
    SpeakerConfusionRate,
)

class Test:
    def __init__(self, test_dataset, model, step=2.5):

        self.test_dataset = test_dataset
        self.model = model
        (self.device,) = get_devices(needs=1)
        self.inference = Inference(self.model, step=step, device=self.device)

        self.sample_rate = test_dataset[0]["audio"]["sampling_rate"]

        # Get the number of frames associated to a chunk:
        _, self.num_frames, _ = self.inference.model(
            torch.rand((1, int(self.inference.duration * self.sample_rate))).to(
                self.device
            )
        ).shape
        # compute frame resolution:
        self.resolution = self.inference.duration / self.num_frames

        self.metrics = {
            "der": DiarizationErrorRate(0.5).to(self.device),
            "confusion": SpeakerConfusionRate(0.5).to(self.device),
            "missed_detection": MissedDetectionRate(0.5).to(self.device),
            "false_alarm": FalseAlarmRate(0.5).to(self.device),
        }

    def predict(self, file):
        audio = (
            torch.tensor(file["audio"]["array"])
            .unsqueeze(0)
            .to(torch.float32)
            .to(self.device)
        )
        sample_rate = file["audio"]["sampling_rate"]

        input = {"waveform": audio, "sample_rate": sample_rate}

        prediction = self.inference(input)

        return prediction

    def compute_gt(self, file):

        audio = torch.tensor(file["audio"]["array"]).unsqueeze(0).to(torch.float32)
        sample_rate = file["audio"]["sampling_rate"]

        audio_duration = len(audio[0]) / sample_rate
        num_frames = int(round(audio_duration / self.resolution))

        labels = list(set(file["speakers"]))

        gt = np.zeros((num_frames, len(labels)), dtype=np.uint8)

        for i in range(len(file["timestamps_start"])):
            start = file["timestamps_start"][i]
            end = file["timestamps_end"][i]
            speaker = file["speakers"][i]
            start_frame = int(round(start / self.resolution))
            end_frame = int(round(end / self.resolution))
            speaker_index = labels.index(speaker)

            gt[start_frame:end_frame, speaker_index] += 1

        return gt

    def compute_metrics_on_file(self, file):

        gt = self.compute_gt(file)
        prediction = self.predict(file)

        sliding_window = SlidingWindow(
            start=0, step=self.resolution, duration=self.resolution
        )
        labels = list(set(file["speakers"]))

        reference = SlidingWindowFeature(
            data=gt, labels=labels, sliding_window=sliding_window
        )

        for window, pred in prediction:

            reference_window = reference.crop(window, mode="center")
            common_num_frames = min(self.num_frames, reference_window.shape[0])

            ref_num_frames, ref_num_speakers = reference_window.shape
            pred_num_frames, pred_num_speakers = pred.shape

            if pred_num_speakers > ref_num_speakers:
                reference_window = np.pad(
                    reference_window, ((0, 0), (0, pred_num_speakers - ref_num_speakers))
                )
            elif ref_num_speakers > pred_num_speakers:
                pred = np.pad(
                    pred, ((0, 0), (0, ref_num_speakers - pred_num_speakers))
                )

            pred = (
                torch.tensor(pred[:common_num_frames])
                .unsqueeze(0)
                .permute(0, 2, 1)
                .to(self.device)
            )
            target = (
                torch.tensor(reference_window[:common_num_frames])
                .unsqueeze(0)
                .permute(0, 2, 1)
            ).to(self.device)

            self.metrics["der"](pred, target)
            self.metrics["false_alarm"](pred, target)
            self.metrics["missed_detection"](pred, target)
            self.metrics["confusion"](pred, target)
        

    def compute_metrics(self):

        for file in tqdm(self.test_dataset):
            self.compute_metrics_on_file(file)

        return {
            "der": self.metrics["der"].compute(),
            "false_alarm": self.metrics["false_alarm"].compute(),
            "missed_detection": self.metrics["missed_detection"].compute(),
            "confusion": self.metrics["confusion"].compute(),
        }

