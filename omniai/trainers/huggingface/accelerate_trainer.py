import logging
from typing import Optional, List, Union

import datasets
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ray.air import Result
from ray.air.config import ScalingConfig, RunConfig
from ray.train.huggingface.accelerate import AccelerateTrainer as RayAccelerateTrainer

from omniai.cli import OmniSession
from omniai.models import HFCausalLM
from omniai.trainers import BaseTrainer

logger = get_logger(__name__)


def train_loop_per_worker(omni_session: OmniSession) -> None:
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if omni_session.seed is not None:
        set_seed(omni_session.seed)

    datamodule = omni_session.data

    train_dataloader = datamodule.get_train_dataloader()
    val_dataloader = datamodule.get_val_dataloader()

    model: HFCausalLM = omni_session.model
    tokenizer = model.tokenizer

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()


class AccelerateTrainer(RayAccelerateTrainer, BaseTrainer):
    def __init__(
        self,
        accelerate_config: Optional[Union[dict, str]] = None,
        scaling_config: Optional[ScalingConfig] = None,
        run_config: Optional[RunConfig] = None,
        split_batches: bool = False,
        mixed_precision: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
        log_with: Optional[Union[str, List[str]]] = None,
        project_dir: Optional[str] = None,
        even_batches: bool = True,
    ) -> None:
        RayAccelerateTrainer.__init__(
            self,
            train_loop_per_worker=train_loop_per_worker,
            accelerate_config=accelerate_config,
            scaling_config=scaling_config,
            run_config=run_config,
        )

    def fit(self, session: OmniSession) -> Result:
        self._train_loop_config = session
        return RayAccelerateTrainer.fit(self)
