from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchok.constructor.constructor import Constructor

from .group_sampler import GroupSampler


class ConstructorWithCustomSampler(Constructor):
    @staticmethod
    def _prepare_dataloader(dataset_params: DictConfig, dataloader_params: DictConfig) -> DataLoader:
        dataset = Constructor._create_dataset(dataset_params)
        collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None

        ## additions to src code in torchok
        batch_sampler = GroupSampler(
            dataset,
            batch_size=dataloader_params['batch_size'],
            drop_last=dataloader_params['drop_last']
        )
        loader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            collate_fn=collate_fn,
                            **dataloader_params)

        return loader
