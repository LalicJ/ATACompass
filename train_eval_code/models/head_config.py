from transformers import PretrainedConfig
from typing import List


class HeadConfig(PretrainedConfig):
    # model_type = "resnet"

    def __init__(
        self,
        hyena_model_config,
        head_config,
        num_seqs,
        **kwargs
    ):
        self.hyena_model_config = hyena_model_config
        self.head_config = head_config
        self.num_seqs = num_seqs

        super().__init__(**kwargs)
