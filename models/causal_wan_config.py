# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from models.dits_base import DiTArchConfig, DiTConfig


def is_causal_block(n: str, m) -> bool:
    parts = n.split(".")
    return len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit()


@dataclass
class CausalWanArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_causal_block]
    )


@dataclass
class CausalWanConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=CausalWanArchConfig)

    prefix: str = "CausalWan"
