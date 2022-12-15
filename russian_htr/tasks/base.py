from omegaconf import DictConfig
from torchok.tasks.base import BaseTask

from .constructor import ConstructorWithCustomSampler


class BaseTaskWithCustomConstructor(BaseTask):
    def __init__(self, hparams: DictConfig, inputs=None, **kwargs):
        """Init BaseTask.
        Args:
            hparams: Hyperparameters that set in yaml file.
            inputs: information about input model shapes and dtypes.
        """
        super().__init__(hparams=hparams, inputs=inputs, **kwargs)
        self._constructor = ConstructorWithCustomSampler(hparams)
