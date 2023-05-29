from dataclasses import dataclass
from typing import Optional

import ray
import torch.nn as nn
from jsonargparse import CLI
from ray.train.base_trainer import BaseTrainer

from omniai.data.datamodule import DataModule


@dataclass
class RayClusterConfig:
    # Starts Ray on local machine.
    start_local: bool = False

    # Ray address to connect to.
    address: str = "auto"

    # Number of workers to use.
    num_workers: int = 1


def fit(
    *,
    ray_cluster: Optional[RayClusterConfig] = None,
    data: DataModule,
    model: nn.Module,
    trainer: BaseTrainer,
):
    if ray_cluster is None:
        ray_cluster = RayClusterConfig()

    if ray_cluster.start_local:
        # Start a local Ray runtime.
        ray.init(num_cpus=ray_cluster.num_workers + 2)
    else:
        # Connect to a Ray cluster for distributed training.
        ray.init(address=ray_cluster.address)

    results = trainer.fit()
    print(results)


if __name__ == "__main__":
    CLI(
        components=[fit],
        as_positional=False
    )
