import contextlib
from typing import Optional

import torch.nn as nn

from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


class HFCausalLM(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        no_init: bool = True,
    ):
        super().__init__()

        if no_init:
            init_ctx = init_empty_weights()
        else:
            init_ctx = contextlib.nullcontext()

        with init_ctx:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_name_or_path
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        return self.model.resize_token_embeddings(new_num_tokens)

    def tie_weights(self) -> None:
        return self.model.tie_weights()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
