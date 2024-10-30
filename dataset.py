# dataset.py
import torch
from torch.utils.data import Dataset
import random
from typing import List, Dict
from pathlib import Path
import numpy as np

class RawTextDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,
        max_length: int = 256,
        mlm_probability: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.examples = []
        
        # Load and process all text files
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                self.examples.extend(self._process_text(text))
    
    def _process_text(self, text: str) -> List[Dict[str, torch.Tensor]]:
        """Process raw text into chunks of max_length tokens."""
        tokenized = self.tokenizer.encode(text)
        
        chunks = []
        for i in range(0, len(tokenized.ids), self.max_length):
            chunk_ids = tokenized.ids[i:i + self.max_length]
            if len(chunk_ids) < 10:  # Skip very short chunks
                continue
                
            # Pad if necessary
            if len(chunk_ids) < self.max_length:
                chunk_ids.extend([self.tokenizer.token_to_id("<pad>")] * 
                               (self.max_length - len(chunk_ids)))
            
            chunks.append({
                "input_ids": torch.tensor(chunk_ids, dtype=torch.long),
                "attention_mask": torch.ones(self.max_length, dtype=torch.long)
            })
        return chunks
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        inputs = self._mask_tokens(example["input_ids"])
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": inputs["labels"]
        }
    
    def _mask_tokens(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create masked tokens for MLM training."""
        labels = inputs.clone()
        
        # Create probability mask using numpy instead of torch.bernoulli
        probability_matrix = np.random.random(size=inputs.shape)
        probability_matrix = probability_matrix < self.mlm_probability
        
        # Convert to tensor
        masked_indices = torch.tensor(probability_matrix, dtype=torch.bool)
        
        # Don't mask padding tokens
        pad_token_id = self.tokenizer.token_to_id("<pad>")
        masked_indices[inputs == pad_token_id] = False
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = np.random.random(size=inputs.shape) < 0.8
        indices_replaced = torch.tensor(indices_replaced, dtype=torch.bool) & masked_indices
        
        mask_token_id = self.tokenizer.token_to_id("<mask>")
        inputs[indices_replaced] = mask_token_id
        
        return {"input_ids": inputs, "labels": labels}