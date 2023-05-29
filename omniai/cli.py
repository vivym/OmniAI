import random
import importlib
from typing import Any, Dict, Optional, List, Union, Tuple

import torch.nn as nn
from jsonargparse import ArgumentParser, ActionConfigFile, Namespace
from jsonargparse.actions import _ActionSubCommands
from torch.optim import Optimizer

from omniai.data import DataModule
from omniai.trainers import BaseTrainer
from omniai.utils.torch import TORCH_GREATER_EQUAL_2_0

if TORCH_GREATER_EQUAL_2_0:
    from torch.optim.lr_scheduler import LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})

    if not isinstance(args, tuple):
        args = (args,)

    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = importlib.import_module(class_module)
    cls = getattr(module, class_name)

    return cls(*args, **kwargs)


class OmniArgumentParser(ArgumentParser):
    def __init__(
        self,
        *args: Any,
        description: str = "OmniAI CLI",
        env_prefix: str = "OMNIAI",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args, description=description, env_prefix=env_prefix, **kwargs
        )


class OmniSession:
    def __init__(
        self,
        raw_config: Namespace,
    ) -> None:
        self.raw_config = raw_config

        self._data = None
        self._model = None
        self._optimizer = None
        self._lr_scheduler = None

        seed = self.raw_config["seed"]
        if isinstance(seed, bool):
            if seed:
                self.seed = random.randint(0, 2 ** 30)
            else:
                self.seed = None
        else:
            assert isinstance(seed, int)
            self.seed = seed

    @property
    def data(self) -> DataModule:
        if self._data is None:
            self._data = instantiate_class((), init=self.raw_config["data"])

        return self._data

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = instantiate_class((), init=self.raw_config["model"])

        return self._model

    @property
    def optimizer(self) -> Optimizer:
        if self._optimizer is None:
            self._optimizer = instantiate_class(
                self.model.parameters(), init=self.raw_config["optimizer"]
            )

        return self._optimizer

    @property
    def lr_scheduler(self) -> Optional[LRScheduler]:
        if self._lr_scheduler is None:
            if "lr_scheduler" in self.raw_config:
                self._lr_scheduler = instantiate_class(
                    self.optimizer, init=self.raw_config["lr_scheduler"]
                )

        return self._lr_scheduler


class OmniCLI:
    """OmniAI CLI"""

    def __init__(
        self,
        args: Optional[List[str]] = None,
    ) -> None:
        self.parser = self.init_parser()

        subcommands = self.parser.add_subcommands(required=True)
        self.add_subcommand(
            "fit",
            help="Full training pipeline",
            subcommands=subcommands,
        )
        self.add_subcommand(
            "validate",
            help="Validation pipeline",
            subcommands=subcommands,
        )
        self.add_subcommand(
            "test",
            help="Test pipeline",
            subcommands=subcommands,
        )
        self.add_subcommand(
            "predict",
            help="Prediction pipeline",
            subcommands=subcommands,
        )

        self.config = self.parser.parse_args(args)

        # Instantiate
        self.config_instantiated = self.parser.instantiate_classes(self.config)

        subcommand = self.config["subcommand"]
        self.trainer = self.config_instantiated[subcommand]["trainer"]

        # Run subcommand
        session = OmniSession(self.config_instantiated[subcommand])
        getattr(self.trainer, subcommand)(session)

    def add_subcommand(
        self,
        name: str,
        help: str,
        subcommands: _ActionSubCommands,
        include_optimizer: bool = True,
    ) -> None:
        parser = self.init_parser()
        parser.add_argument(
            "--seed",
            type=Union[bool, int],
            default=True,
            help=(
                "Random seed. "
                "If True, a random seed will be generated. "
                "If False, no random seed will be used. "
                "If an integer, that integer will be used as the random seed."
            ),
        )

        parser.add_subclass_arguments(
            DataModule,
            nested_key="data",
            required=True,
            fail_untyped=False,
            instantiate=False,
        )

        parser.add_subclass_arguments(
            nn.Module,
            nested_key="model",
            required=True,
            fail_untyped=False,
            instantiate=False,
        )

        if include_optimizer:
            parser.add_subclass_arguments(
                Optimizer,
                nested_key="optimizer",
                required=True,
                fail_untyped=False,
                instantiate=False,
                skip={"params"},
            )
            parser.add_subclass_arguments(
                LRScheduler,
                nested_key="lr_scheduler",
                required=False,
                fail_untyped=False,
                instantiate=False,
                skip={"optimizer"},
            )

        parser.add_subclass_arguments(
            BaseTrainer,
            nested_key="trainer",
            required=True,
            fail_untyped=False,
        )

        subcommands.add_subcommand(name, parser, help=help)

    def init_parser(self) -> OmniArgumentParser:
        parser = OmniArgumentParser()
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        return parser


if __name__ == "__main__":
    OmniCLI()
