# tokenizer_trainer.py
import os
import glob
from typing import List, Dict
from pathlib import Path
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor

class TokenizerTrainer:
    def __init__(self, vocab_size: int = 50000, min_frequency: int = 2):
        # Initialize a BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # Add byte-level pre-tokenization
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.pre_tokenizer.add_prefix_space=True
        
        # Add byte-level post-processing
        self.tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
        
        # Configure the trainer
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[
                "<pad>",
                "<s>",
                "</s>",
                "<unk>",
                "<mask>"
            ],
            show_progress=True
        )
        
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train(self, files_path, output_dir: Path):
        """Train the tokenizer on the provided files."""
        # Convert input to list if it's a single file/directory
        if isinstance(files_path, (str, Path)):
            if os.path.isdir(files_path):
                training_files = glob.glob(os.path.join(files_path, "*.txt"))
            else:
                training_files = [files_path]
        elif isinstance(files_path, list):
            training_files = files_path
        else:
            raise ValueError("files_path must be a string, Path, or list of files")

        if not training_files:
            raise ValueError(f"No training files found in {files_path}")

        # Train the tokenizer
        self.tokenizer.train(training_files, self.trainer)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the tokenizer and configurations
        self.save(output_dir)
    
    def save(self, output_dir: Path):
        """Save tokenizer and its configuration files."""
        # Save the main tokenizer
        self.tokenizer.save(str(output_dir / "tokenizer.json"))
        
        # Save tokenizer config
        tokenizer_config = {
            "vocab_size": self.vocab_size,
            "padding_side": "right",
            "truncation_side": "right",
            "model_max_length": 256,
            "model_type": "bpe",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
            "clean_up_tokenization_spaces": True
        }
        
        with open(output_dir / "tokenizer_config.json", 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # Save special tokens map
        special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        }
        
        with open(output_dir / "special_tokens_map.json", 'w') as f:
            json.dump(special_tokens_map, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, directory: Path):
        """Load a pretrained tokenizer from a directory."""
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        return tokenizer

def train_tokenizer(
    input_dir: str,
    output_dir: str,
    vocab_size: int = 50000,
    min_frequency: int = 2
):
    """Convenience function to train a tokenizer."""
    trainer = TokenizerTrainer(vocab_size=vocab_size, min_frequency=min_frequency)
    trainer.train(input_dir, Path(output_dir))
    print(f"Tokenizer trained and saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing training text files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save the tokenizer files')
    parser.add_argument('--vocab_size', type=int, default=50000,
                       help='Size of the vocabulary')
    parser.add_argument('--min_frequency', type=int, default=2,
                       help='Minimum frequency for a token to be included')
    
    args = parser.parse_args()
    
    train_tokenizer(
        args.input_dir,
        args.output_dir,
        args.vocab_size,
        args.min_frequency
    )