# test_model.py
import os
from pathlib import Path
from tokenizerTrainer import TokenizerTrainer
from config import ELECTRAConfig
from model import ELECTRA
import torch
from torch.utils.data import DataLoader
from dataset import RawTextDataset
import time
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device():
    """Get the appropriate device (MPS for Mac, CUDA for NVIDIA, or CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def setup_test():
    # Create test directory structure
    test_dir = Path("test_electra")
    test_dir.mkdir(exist_ok=True)
    
    # Create a small test text file
    test_text = """This is a sample text for testing the ELECTRA model.
    It contains multiple sentences to test tokenization and training.
    The model should learn to distinguish real from fake tokens."""
    
    with open(test_dir / "test.txt", "w") as f:
        f.write(test_text)
    
    return test_dir

def print_dimensions(model, batch):
    """Helper function to print dimensions of tensors at each step."""
    print("\nDimensions check:")
    with torch.no_grad():
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        # Generator embeddings
        emb = model.generator.embeddings(batch['input_ids'])
        print(f"Generator embeddings shape: {emb.shape}")
        
        # Generator encoder output
        enc = model.generator.encoder(emb)
        print(f"Generator encoder output shape: {enc.shape}")
        
        try:
            # Generator predictions
            pred = model.generator.generator_predictions(enc)
            print(f"Generator predictions shape: {pred.shape}")
        except Exception as e:
            print(f"Error in generator predictions: {e}")

def train_test_pipeline():
    # Setup test environment
    test_dir = setup_test()
    tokenizer_dir = test_dir / "tokenizer"
    model_dir = test_dir / "model"
    tokenizer_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Train tokenizer
    print("Training tokenizer...")
    trainer = TokenizerTrainer(vocab_size=1000)  # Small vocab for testing
    trainer.train(str(test_dir), tokenizer_dir)
    tokenizer = trainer.tokenizer

    # Initialize model with small config for testing
    print("Initializing model...")
    config = ELECTRAConfig(
        vocab_size=1000,
        max_position_embeddings=256,
        hidden_size=256,  # Make this larger and divisible by num_attention_heads
        num_hidden_layers=2,
        num_attention_heads=8,  # Must divide hidden_size evenly
        intermediate_size=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        generator_size_divisor=4
    )

    # Modify dataset creation
    dataset = RawTextDataset(
        file_paths=[str(test_dir / "test.txt")],
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0  # Important for MPS compatibility
    )

    # Add dimension checking at the start
    device = get_device()
    model = ELECTRA(config)
    model.to(device)
    
    # Print model structure
    print("\nModel structure:")
    print(f"Generator hidden size: {config.generator_config['hidden_size']}")
    print(f"Discriminator hidden size: {config.hidden_size}")
    
    # Test a forward pass with a small batch
    test_batch = next(iter(dataloader))
    test_batch = {k: v.to(device) for k, v in test_batch.items()}
    print_dimensions(model, test_batch)

    # Prepare dataset
    print("Preparing dataset...")
    dataset = RawTextDataset(
        file_paths=[str(test_dir / "test.txt")],
        tokenizer=tokenizer,
        max_length=256
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0  # Important for MPS compatibility
    )

    # Training loop
    print("Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}/2")
        progress_bar = tqdm(dataloader, desc=f"Training")
        
        epoch_loss = 0
        for batch_idx, batch in enumerate(progress_bar):
            try:
                start_time = time.time()
                
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs["loss"]
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                    'batch_time': f'{(time.time() - start_time):.2f}s'
                })
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue
        
        print(f"\nEpoch {epoch+1} average loss: {epoch_loss/len(dataloader):.4f}")

    # Save model
    print("Saving model...")
    model.save_pretrained(model_dir)
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        try:
            test_input = "This is a test"
            encoded = tokenizer.encode(test_input)
            input_ids = torch.tensor([encoded.ids]).to(device)
            
            # Ensure input shape is correct
            if input_ids.size(1) < config.max_position_embeddings:
                # Pad to max length
                pad_length = config.max_position_embeddings - input_ids.size(1)
                pad_token_id = tokenizer.token_to_id("<pad>")
                padding = torch.full((1, pad_length), pad_token_id, device=device)
                input_ids = torch.cat([input_ids, padding], dim=1)
            
            # Generate tokens using the generator
            outputs = model.generator(input_ids)
            predictions = torch.argmax(outputs, dim=-1)
            
            # Move predictions to CPU before decoding
            predictions = predictions.cpu()
            
            # Decode the predictions
            decoded = tokenizer.decode(predictions[0].tolist())
            print(f"Input: {test_input}")
            print(f"Generated: {decoded}")
            
        except RuntimeError as e:
            print(f"Error in generation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        train_test_pipeline()
    except Exception as e:
        print(f"Training pipeline failed: {e}")