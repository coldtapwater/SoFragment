# config.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
from pathlib import Path

@dataclass
class ELECTRAConfig:
    vocab_size: int = 50000
    max_position_embeddings: int = 256
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 256
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    summary_type: str = "first"
    summary_use_proj: bool = True
    summary_activation: str = "gelu"
    summary_last_dropout: float = 0.1
    pad_token_id: int = 0
    mask_token_id: Optional[int] = None
    generator_size_divisor: int = 4  # Generator will be 1/4 the size of discriminator

    @property
    def generator_config(self) -> Dict[str, Any]:
        """Returns the configuration for the generator component."""
        return {
            "hidden_size": self.hidden_size // self.generator_size_divisor,
            "num_hidden_layers": self.num_hidden_layers // self.generator_size_divisor,
            "num_attention_heads": self.num_attention_heads // self.generator_size_divisor,
            "intermediate_size": self.intermediate_size // self.generator_size_divisor,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "embedding_size": self.hidden_size  # Add this line
        }
    
    def save(self, save_directory: Path):
        config_file = save_directory / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def from_json(cls, file_path: Path):
        with open(file_path) as f:
            return cls(**json.load(f))