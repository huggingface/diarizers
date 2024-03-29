import numpy as np
import torch


class DataCollator:
    """Data collator that will dynamically pad the target labels to have max_speakers_per_chunk"""

    def __init__(self, max_speakers_per_chunk) -> None:

        self.max_speakers_per_chunk = max_speakers_per_chunk

    def __call__(self, features):
        """_summary_

        Args:
            features (_type_): _description_

        Returns:
            _type_: _description_
        """

        batch = {}

        speakers = [f["nb_speakers"] for f in features]
        labels = [f["labels"] for f in features]

        batch["labels"] = self.pad_targets(labels, speakers)

        batch["waveforms"] = torch.stack([f["waveforms"] for f in features])

        return batch

    def pad_targets(self, labels, speakers):
        """
        labels:
        speakers:

        Returns:
            _type_:
                Collated target tensor of shape (num_frames, self.max_speakers_per_chunk)
                If one chunk has more than max_speakers_per_chunk speakers, we keep
                the max_speakers_per_chunk most talkative ones. If it has less, we pad with
                zeros (artificial inactive speakers).
        """

        targets = []

        for i in range(len(labels)):

            label = speakers[i]
            target = labels[i].numpy()
            num_speakers = len(label)

            if num_speakers > self.max_speakers_per_chunk:
                indices = np.argsort(-np.sum(target, axis=0), axis=0)
                target = target[:, indices[: self.max_speakers_per_chunk]]

            elif num_speakers < self.max_speakers_per_chunk:
                target = np.pad(
                    target,
                    ((0, 0), (0, self.max_speakers_per_chunk - num_speakers)),
                    mode="constant",
                )

            targets.append(target)

        return torch.from_numpy(np.stack(targets))
