from itertools import chain
from typing import Any, Dict, List, Optional, Union

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, interleave_datasets, concatenate_datasets, Dataset, IterableDataset
from transformers import AutoTokenizer, default_data_collator, PreTrainedTokenizerBase
from torch.utils.data import DataLoader

from ..datamodule import DataModule

logger = get_logger(__name__)


class DatasetConfig:
    path: str
    sample_rate: float = 1.0
    name: Optional[str] = None
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    use_auth_token: Optional[bool] = None
    streaming: bool = False
    num_proc: Optional[int] = None
    languages: Optional[List[str]] = None
    config_kwargs: Optional[Dict[str, Any]] = None


class LLMPretrainingDataModule(DataModule):
    def __init__(
        self,
        datasets: List[DatasetConfig],
        batch_size: int,
        num_workers: int,
        preprocessing_num_workers: int,
    ) -> None:
        super().__init__()

        self.dataset_configs = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_num_workers = preprocessing_num_workers

    def _load_dataset(
        self, dataset_config: DatasetConfig, split: str
    ) -> Union[Dataset, IterableDataset]:
        config_kwargs = dataset_config.config_kwargs or {}
        if dataset_config.languages is not None:
            config_kwargs["languages"] = dataset_config.languages

        return load_dataset(
            dataset_config.path,
            split=split,
            name=dataset_config.name,
            cache_dir=dataset_config.cache_dir,
            revision=dataset_config.revision,
            use_auth_token=dataset_config.use_auth_token,
            streaming=dataset_config.streaming,
            num_proc=dataset_config.num_proc,
            **config_kwargs,
        )

    def apply_preprocessing(
        self,
        dataset: Union[Dataset, IterableDataset],
        accelerator: Accelerator,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Union[Dataset, IterableDataset]:
        column_names = dataset.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        with accelerator.main_process_first():
            tokenized_dataset = dataset.map(
                lambda samples: tokenizer(samples[text_column_name]),
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
            )

        block_size = min(tokenizer.model_max_length, 1024)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size
        def group_texts(samples: Dict):
            # Concatenate all texts.
            concatenated_samples = {
                k: list(chain(*samples[k])) for k in samples.keys()
            }
            total_length = len(concatenated_samples[list(samples.keys())[0]])

            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
                for k, t in concatenated_samples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.

        with accelerator.main_process_first():
            grouped_dataset = tokenized_dataset.map(
                group_texts,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        return grouped_dataset

    def get_train_dataloader(
        self,
        accelerator: Accelerator,
        tokenizer: PreTrainedTokenizerBase,
    ):
        datasets = [
            load_dataset(dataset_config, split="train")
            for dataset_config in self.dataset_configs
        ]
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = interleave_datasets(datasets)

        dataset = self.apply_preprocessing(dataset, accelerator, tokenizer)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=default_data_collator,
        )

    def get_val_dataloader(
        self,
        accelerator: Accelerator,
        tokenizer: PreTrainedTokenizerBase,
    ):
        datasets = [
            load_dataset(dataset_config, split="validation")
            for dataset_config in self.dataset_configs
        ]
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = concatenate_datasets(datasets)

        dataset = self.apply_preprocessing(dataset, accelerator, tokenizer)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=default_data_collator,
        )

    def get_test_dataloader(
        self,
        accelerator: Accelerator,
        tokenizer: PreTrainedTokenizerBase,
    ):
        datasets = [
            load_dataset(dataset_config, split="test")
            for dataset_config in self.dataset_configs
        ]
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = concatenate_datasets(datasets)

        dataset = self.apply_preprocessing(dataset, accelerator, tokenizer)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=default_data_collator,
        )
