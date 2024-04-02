import copy
from typing import Optional

import torch
from diarizers.models.segmentation.pyannet import PyanNet_nn
from transformers import PretrainedConfig, PreTrainedModel

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.utils.loss import binary_cross_entropy, nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset


class SegmentationModelConfig(PretrainedConfig):
    """Config class associated with SegmentationModel model."""

    model_type = "pyannet"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class SegmentationModel(PreTrainedModel):
    """
    Wrapper class for the PyanNet segmentation model used in pyannote.
    Inherits from Pretrained model to be compatible with the HF Trainer.
    Can be used to train segmentation models adapted for the "SpeakerDiarisation Task" in pyannote.
    """

    config_class = SegmentationModelConfig

    def __init__(
        self,
        config=SegmentationModelConfig(), 
        hyperparameters = {
            'chunk_duration' : 10, 
            'max_speakers_per_frame' : 2, 
            'max_speakers_per_chunk': 3, 
            'min_duration' : None, 
            'warm_up' : (0.0, 0.0), 
            'weigh_by_cardinality': False
        }, 
    ):
        """init method

        Args:
            config (_type_): instance of SegmentationModelConfig.
            min_duration (_type_, optional): _description_. Defaults to None.
            duration : float, optional
                    Chunks duration processed by the model. Defaults to 10s.
            max_speakers_per_chunk : int, optional
                Maximum number of speakers per chunk.
            max_speakers_per_frame : int, optional
                Maximum number of (overlapping) speakers per frame.
                Setting this value to 1 or more enables `powerset multi-class` training.
                Default behavior is to use `multi-label` training.
            weigh_by_cardinality: bool, optional                NOT IMPPLEMENTED HERE!
                Weigh each powerset classes by the size of the corresponding speaker set.
                In other words, {0, 1} powerset class weight is 2x bigger than that of {0}
                or {1} powerset classes. Note that empty (non-speech) powerset class is
                assigned the same weight as mono-speaker classes. Defaults to False (i.e. use
                same weight for every class). Has no effect with `multi-label` training.

            min_duration : float, optional                      NOT IMPLEMENTED HERE!
                Sample training chunks duration uniformely between `min_duration`
                and `duration`. Defaults to `duration` (i.e. fixed length chunks).
            warm_up : float or (float, float), optional
                Use that many seconds on the left- and rightmost parts of each chunk
                to warm up the model. While the model does process those left- and right-most
                parts, only the remaining central part of each chunk is used for computing the
                loss during training, and for aggregating scores during inference.
                Defaults to 0. (i.e. no warm-up).
        """

        super().__init__(config)
        self.model = PyanNet_nn(sincnet={"stride": 10})

        self.weigh_by_cardinality = hyperparameters['weigh_by_cardinality']
        self.max_speakers_per_frame = hyperparameters['max_speakers_per_frame']
        self.chunk_duration = hyperparameters['chunk_duration']
        self.min_duration = hyperparameters['min_duration']
        self.warm_up = hyperparameters['warm_up']
        self.max_speakers_per_chunk = hyperparameters['max_speakers_per_chunk']

        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION
            if self.max_speakers_per_frame is None
            else Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.chunk_duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
            permutation_invariant=True,
        )
        self.model.specifications = self.specifications
        self.model.build()
        self.setup_loss_func()

    def forward(self, waveforms, labels=None, nb_speakers=None):
        """foward pass of the Pretrained Model.

        Args:
            waveforms (_type_): _description_
            labels (_type_, optional): _description_. Defaults to None.
            nb_speakers (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        prediction = self.model(waveforms.unsqueeze(1))
        batch_size, num_frames, _ = prediction.shape

        if labels is not None:

            weight = torch.ones(batch_size, num_frames, 1, device=waveforms.device)
            warm_up_left = round(
                self.specifications.warm_up[0]
                / self.specifications.duration
                * num_frames
            )
            weight[:, :warm_up_left] = 0.0
            warm_up_right = round(
                self.specifications.warm_up[1]
                / self.specifications.duration
                * num_frames
            )
            weight[:, num_frames - warm_up_right :] = 0.0

            if self.specifications.powerset:

                multilabel = self.model.powerset.to_multilabel(prediction)
                permutated_target, _ = permutate(multilabel, labels)

                permutated_target_powerset = self.model.powerset.to_powerset(
                    permutated_target.float()
                )
                loss = self.segmentation_loss(
                    prediction, permutated_target_powerset, weight=weight
                )

            else:
                permutated_prediction, _ = permutate(labels, prediction)
                loss = self.segmentation_loss(
                    permutated_prediction, labels, weight=weight
                )

            return {"loss": loss, "logits": prediction}

        return {"logits": prediction}

    def setup_loss_func(self):
        """setup the loss function is self.specifications.powerset is True."""
        if self.specifications.powerset:
            self.model.powerset = Powerset(
                len(self.specifications.classes),
                self.specifications.powerset_max_classes,
            )

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
            Permutated speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        seg_loss : torch.Tensor
            Permutation-invariant segmentation loss
        """

        if self.specifications.powerset:
            # `clamp_min` is needed to set non-speech weight to 1.
            class_weight = (
                torch.clamp_min(self.model.powerset.cardinality, 1.0)
                if self.weigh_by_cardinality
                else None
            )
            seg_loss = nll_loss(
                permutated_prediction,
                torch.argmax(target, dim=-1),
                class_weight=class_weight,
                weight=weight,
            )
        else:
            seg_loss = binary_cross_entropy(
                permutated_prediction, target.float(), weight=weight
            )

        return seg_loss

    def from_pyannote_model(self, pretrained):
        """Copy the weights and architecture of a pre-trained Pyannote model.

        Args:
            pretrained (pyannote.core.Model): pretrained pyannote segmentation model.
        """

        self.model.hparams = copy.deepcopy(pretrained.hparams)

        self.model.sincnet = copy.deepcopy(pretrained.sincnet)
        self.model.sincnet.load_state_dict(pretrained.sincnet.state_dict())

        self.model.lstm = copy.deepcopy(pretrained.lstm)
        self.model.lstm.load_state_dict(pretrained.lstm.state_dict())

        self.model.linear = copy.deepcopy(pretrained.linear)
        self.model.linear.load_state_dict(pretrained.linear.state_dict())

        self.model.specifications = copy.deepcopy(pretrained.specifications)

        self.model.classifier = copy.deepcopy(pretrained.classifier)
        self.model.classifier.load_state_dict(pretrained.classifier.state_dict())

        self.model.activation = copy.deepcopy(pretrained.activation)
        self.model.activation.load_state_dict(pretrained.activation.state_dict())

        self.specifications = self.model.specifications
        self.model.build()
        self.setup_loss_func()

    def to_pyannote_model(self):
        """Convert the current model to a pyanote segmentation model for use in pyannote pipelines."""

        seg_model = PyanNet(sincnet={"stride": 10})
        seg_model.hparams.update(self.model.hparams)

        seg_model.sincnet = copy.deepcopy(self.model.sincnet)
        seg_model.sincnet.load_state_dict(self.model.sincnet.state_dict())

        seg_model.lstm = copy.deepcopy(self.model.lstm)
        seg_model.lstm.load_state_dict(self.model.lstm.state_dict())

        seg_model.linear = copy.deepcopy(self.model.linear)
        seg_model.linear.load_state_dict(self.model.linear.state_dict())

        seg_model.classifier = copy.deepcopy(self.model.classifier)
        seg_model.classifier.load_state_dict(self.model.classifier.state_dict())

        seg_model.activation = copy.deepcopy(self.model.activation)
        seg_model.activation.load_state_dict(self.model.activation.state_dict())

        seg_model.specifications = self.specifications

        return seg_model
