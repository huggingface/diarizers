# Adapted from: https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/core/inference.py
# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
from pyannote.audio import Inference
from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.torchmetrics import (DiarizationErrorRate, FalseAlarmRate,
                                         MissedDetectionRate,
                                         SpeakerConfusionRate)
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.metrics import diarization
from tqdm import tqdm


class Test:
    """Class used to evaluate a SegmentationModel at inference time on a given test set."""

    def __init__(self, test_dataset, model, step=2.5):
        """init method

        Args:
            test_dataset (_type_): Hugging Face speaker diarization test dataset
            model (SegmentationModel): SegmentationModel used at inference.
            step (float, optional): Steps between successive generated audio chunks. Defaults to 2.5.
        """

        self.test_dataset = test_dataset
        self.model = model
        (self.device,) = get_devices(needs=1)
        self.inference = Inference(self.model, step=step, device=self.device)

        self.sample_rate = test_dataset[0]["audio"]["sampling_rate"]

        # Get the number of frames associated to a chunk:
        _, self.num_frames, _ = self.inference.model(
            torch.rand((1, int(self.inference.duration * self.sample_rate))).to(self.device)
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
        """Make a prediction on a dataset row,
            using pyannote inference object.

        Args:
            file (_type_): _description_

        Returns:
            _type_: _description_
        """
        audio = torch.tensor(file["audio"]["array"]).unsqueeze(0).to(torch.float32).to(self.device)
        sample_rate = file["audio"]["sampling_rate"]

        input = {"waveform": audio, "sample_rate": sample_rate}

        prediction = self.inference(input)

        return prediction

    def compute_gt(self, file):
        """
        Args:
            file (_type_): dataset row.

        Returns:
            gt: numpy array with shape (num_frames, num_speakers).
        """

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
        """coppute and update metrics on a dataset row.

        Args:
            file (_type_): a Hugging Face dataset row.
        """

        gt = self.compute_gt(file)
        prediction = self.predict(file)

        sliding_window = SlidingWindow(start=0, step=self.resolution, duration=self.resolution)
        labels = list(set(file["speakers"]))

        reference = SlidingWindowFeature(data=gt, labels=labels, sliding_window=sliding_window)

        for window, pred in prediction:
            reference_window = reference.crop(window, mode="center")
            common_num_frames = min(self.num_frames, reference_window.shape[0])

            _, ref_num_speakers = reference_window.shape
            _, pred_num_speakers = pred.shape

            if pred_num_speakers > ref_num_speakers:
                reference_window = np.pad(reference_window, ((0, 0), (0, pred_num_speakers - ref_num_speakers)))
            elif ref_num_speakers > pred_num_speakers:
                pred = np.pad(pred, ((0, 0), (0, ref_num_speakers - pred_num_speakers)))

            pred = torch.tensor(pred[:common_num_frames]).unsqueeze(0).permute(0, 2, 1).to(self.device)
            target = (torch.tensor(reference_window[:common_num_frames]).unsqueeze(0).permute(0, 2, 1)).to(self.device)

            self.metrics["der"](pred, target)
            self.metrics["false_alarm"](pred, target)
            self.metrics["missed_detection"](pred, target)
            self.metrics["confusion"](pred, target)

    def compute_metrics(self):
        """Main method, used to compute speaker diarization metrics on test_dataset.
        Returns:
            dict: metric values.
        """

        for file in tqdm(self.test_dataset):
            self.compute_metrics_on_file(file)

        return {
            "der": self.metrics["der"].compute(),
            "false_alarm": self.metrics["false_alarm"].compute(),
            "missed_detection": self.metrics["missed_detection"].compute(),
            "confusion": self.metrics["confusion"].compute(),
        }


class TestPipeline:
    def __init__(self, test_dataset, pipeline) -> None:

        self.test_dataset = test_dataset

        (self.device,) = get_devices(needs=1)
        self.pipeline = pipeline.to(self.device)
        self.sample_rate = test_dataset[0]["audio"]["sampling_rate"]

        # Get the number of frames associated to a chunk:
        _, self.num_frames, _ = self.pipeline._segmentation.model(
            torch.rand((1, int(self.pipeline._segmentation.duration * self.sample_rate))).to(self.device)
        ).shape
        # compute frame resolution:
        self.resolution = self.pipeline._segmentation.duration / self.num_frames

        self.metrics = {
            "der": diarization.DiarizationErrorRate(),
        }

    def compute_gt(self, file):

        """
        Args:
            file (_type_): dataset row.

        Returns:
            gt: numpy array with shape (num_frames, num_speakers).
        """

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

    def predict(self, file):

        sample = {}
        sample["waveform"] = (
            torch.from_numpy(file["audio"]["array"])
            .to(self.device, dtype=self.pipeline._segmentation.model.dtype)
            .unsqueeze(0)
        )
        sample["sample_rate"] = file["audio"]["sampling_rate"]

        prediction = self.pipeline(sample)

        return prediction

    def compute_metrics_on_file(self, file):

        pred = self.predict(file)
        gt = self.compute_gt(file)

        sliding_window = SlidingWindow(start=0, step=self.resolution, duration=self.resolution)
        gt = SlidingWindowFeature(data=gt, sliding_window=sliding_window)

        gt = self.pipeline.to_annotation(
            gt,
            min_duration_on=0.0,
            min_duration_off=self.pipeline.segmentation.min_duration_off,
        )

        mapping = {label: expected_label for label, expected_label in zip(gt.labels(), self.pipeline.classes())}

        gt = gt.rename_labels(mapping=mapping)

        der = self.metrics["der"](pred, gt)

        return der

    def compute_metrics(self):

        der = 0
        for file in tqdm(self.test_dataset):
            der += self.compute_metrics_on_file(file)

        der /= len(self.test_dataset)

        return {"der": der}
