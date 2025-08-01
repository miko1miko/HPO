
from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Literal, Optional

from trl.trainer.dpo_trainer import DPOConfig as OriginalDPOConfig

@dataclass
class HPOConfig(OriginalDPOConfig):
    """
    Configuration for HPOTrainer, inheriting from trl's DPOConfig.
    Adds specific weights for KL divergence loss on prefix and suffix.
    """

    kl_prefix_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the KL divergence loss on the prefix part."}
    )
    kl_suffix_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the KL divergence loss on the suffix part."}
    )
    mid_dpo_weight: float = field(default=2.0, metadata={"help": "Weight for the DPO loss on the mid (diff) part."})