from typing import Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from torchok.constructor import BACKBONES, HEADS, NECKS, POOLINGS, TASKS
from .base import BaseTaskWithCustomConstructor


@TASKS.register_class
class TextToImageTask(BaseTaskWithCustomConstructor):
    def __init__(
            self,
            hparams: DictConfig,
            backbone_name: str,
            pooling_name: str,
            head_name: str,
            neck_name: str = None,
            backbone_params: dict = None,
            neck_params: dict = None,
            pooling_params: dict = None,
            head_params: dict = None,
            **kwargs
    ):
        """Init ClassificationTask.
        Args:
            hparams: Hyperparameters that set in yaml file.
            TODO: params
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams, **kwargs)

        # BACKBONE
        backbones_params = backbone_params or dict()
        self.backbone = BACKBONES.get(backbone_name)(**backbones_params)

        # NECK
        if neck_name is None:
            self.neck = nn.Identity()
            pooling_in_channels = self.backbone.out_channels
        else:
            neck_params = neck_params or dict()
            self.neck = NECKS.get(neck_name)(in_channels=self.backbone.out_encoder_channels, **neck_params)
            pooling_in_channels = self.neck.out_channels

        # POOLING
        pooling_params = pooling_params or dict()
        self.pooling = POOLINGS.get(pooling_name)(in_channels=pooling_in_channels, **pooling_params)

        # HEAD
        head_params = head_params or dict()
        self.head = HEADS.get(head_name)(in_channels=self.pooling.out_channels, **head_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.backbone(x)
        x = self.neck(x)
        x = self.pooling(x)
        x = self.head(x)
        return x

    def forward_with_gt(self, batch: Dict[str, Union[Tensor, int]]) -> Dict[str, Tensor]:
        """Forward with ground truth labels."""
        input_data = batch.get('image')
        target = batch.get('target')
        features = self.backbone(input_data)
        features = self.neck(features)
        embeddings = self.pooling(features)
        prediction = self.head(embeddings, target)
        output = {'embeddings': embeddings, 'prediction': prediction}

        if target is not None:
            output['target'] = target

        return output

    def as_module(self) -> nn.Sequential:
        """Method for model representation as sequential of modules(need for checkpointing)."""
        return nn.Sequential(self.backbone, self.neck, self.pooling, self.head)