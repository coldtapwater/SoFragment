# example.py
from pathlib import Path
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader
from typing import List
from dataset import RawTextDataset
from model import ELECTRA, ELECTRAConfig
from tqdm import tqdm
import os

def train_example():
    # Initialize config
    config = ELECTRAConfig(
        vocab_size=50000,
        max_position_embeddings=256,
        hidden_size=256,  # Smaller for testing
        num_hidden_layers=6,  # Smaller for testing
        num_attention_heads=8
    )
    
    # Initialize tokenizer
    tokenizer = Tokenizer.from_file("path/to/your/tokenizer.json")
    
    # Initialize model
    model = ELECTRA(config)
    
    # Prepare dataset
    data_files = list(Path("path/to/your/text/files").glob("*.txt"))
    dataset = RawTextDataset(data_files, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
        
        # Save checkpoint
        if save_checkpoint:
            model.save_pretrained("checkpoints")

if __name__ == "__main__":
    train_example()