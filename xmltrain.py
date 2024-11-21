import os
import argparse
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import logging
import time
from datetime import datetime
import glob
import shutil

from tokenizerTrainer import TokenizerTrainer
from config import ELECTRAConfig
from model import ELECTRA

class XMLDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        """
        Dataset for XML-structured conversation data
        Args:
            csv_file: Path to CSV file with 'input' and 'output' columns
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row['Input']
        output_text = row['Output']
        
        # Combine input and output with a separator for context
        combined_text = f"{input_text} [SEP] {output_text}"
        
        # Tokenize the combined text
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by tokenizer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        return item

class XMLTrainer:
    def __init__(
        self,
        train_file: str,
        val_file: str,
        output_dir: str,
        config: ELECTRAConfig = None,
        batch_size: int = 8,
        num_epochs: int = 10,
        learning_rate: float = 2e-4,
        checkpoint_interval: int = 10000,  # Save every 10k steps
        max_checkpoints: int = 5  # Keep only last 5 checkpoints
    ):
        self.train_file = Path(train_file)
        self.val_file = Path(val_file)
        self.output_dir = Path(output_dir)
        self.config = config or self._get_default_config()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()
        
        # Setup device
        self.device = self.get_device()
        self.logger.info(f"Using device: {self.device}")
        
    def _get_default_config(self):
        return ELECTRAConfig(
            vocab_size=50000,
            max_position_embeddings=512,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            generator_size_divisor=4
        )
    
    def setup_directories(self):
        """Create necessary directories for training."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.output_dir / "model"
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.tokenizer_dir = self.output_dir / "tokenizer"
        self.logs_dir = self.output_dir / "logs"
        
        for directory in [
            self.model_dir,
            self.checkpoints_dir,
            self.tokenizer_dir,
            self.logs_dir
        ]:
            directory.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_device(self):
        """Get the appropriate device for training."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    
    def prepare_tokenizer(self):
        """Train tokenizer on the XML dataset."""
        self.logger.info("Preparing tokenizer...")
        
        # Combine input and output text for tokenizer training
        df_train = pd.read_csv(self.train_file)
        df_val = pd.read_csv(self.val_file)
        
        # Create temporary files for tokenizer training
        temp_dir = self.output_dir / "temp_tokenizer_files"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / "combined_text.txt"
        
        try:
            # Combine all text for tokenizer training
            all_text = []
            for df in [df_train, df_val]:
                all_text.extend(df['Input'].tolist())
                all_text.extend(df['Output'].tolist())
            
            # Write combined text to temporary file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_text))
            
            # Train tokenizer
            trainer = TokenizerTrainer(vocab_size=self.config.vocab_size)
            trainer.train([str(temp_file)], self.tokenizer_dir)
            self.tokenizer = trainer.tokenizer
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            self.logger.info("Cleaned up temporary tokenizer training files")
    
    def prepare_data(self):
        """Prepare training and validation datasets."""
        self.logger.info("Preparing datasets...")
        
        self.train_dataset = XMLDataset(
            self.train_file,
            self.tokenizer,
            max_length=self.config.max_position_embeddings
        )
        
        self.val_dataset = XMLDataset(
            self.val_file,
            self.tokenizer,
            max_length=self.config.max_position_embeddings
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def manage_checkpoints(self):
        """Maintain only the specified number of most recent checkpoints."""
        checkpoints = sorted(glob.glob(str(self.checkpoints_dir / "checkpoint_*.pt")))
        while len(checkpoints) > self.max_checkpoints:
            oldest_checkpoint = checkpoints.pop(0)
            os.remove(oldest_checkpoint)
            self.logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, loss):
        """Save a training checkpoint and manage checkpoint history."""
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Manage checkpoint history
        self.manage_checkpoints()
    
    def validate(self, model, val_loader):
        """Run validation and return average loss."""
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs["loss"].item()
        
        avg_loss = total_loss / len(val_loader)
        model.train()
        return avg_loss
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training preparation...")
        
        # Prepare tokenizer and data
        self.prepare_tokenizer()
        self.prepare_data()
        
        # Calculate total steps for progress tracking
        total_steps = len(self.train_loader) * self.num_epochs
        self.logger.info(f"Total training steps: {total_steps:,}")
        
        # Initialize model
        model = ELECTRA(self.config)
        model.to(self.device)
        
        # Setup optimization
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        # Training loop
        self.logger.info("Starting training...")
        best_val_loss = float('inf')
        step = 0
        training_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            model.train()
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in progress_bar:
                try:
                    # Forward pass
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Update tracking
                    epoch_loss += loss.item()
                    step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/step:.4f}'
                    })
                    
                    # Save checkpoint every 10k steps
                    if step % self.checkpoint_interval == 0:
                        self.save_checkpoint(model, optimizer, scheduler, epoch, step, loss.item())
                
                except Exception as e:
                    self.logger.error(f"Error in training step: {str(e)}")
                    continue
            
            # Epoch completion stats
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time/3600:.2f} hours")
            
            # Validation
            val_loss = self.validate(model, self.val_loader)
            self.logger.info(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = self.model_dir / "best_model"
                model.save_pretrained(model_save_path)
                self.logger.info(f"Saved best model to {model_save_path}")
            
            scheduler.step()
        
        # Save final model
        final_model_path = self.model_dir / "final_model"
        model.save_pretrained(final_model_path)
        self.logger.info(f"Saved final model to {final_model_path}")
        
        total_training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_training_time/3600:.2f} hours")
        
        return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train ELECTRA model on XML-structured conversation data')
    
    # Required arguments
    parser.add_argument(
        '--train',
        type=str,
        required=True,
        help='Path to training CSV file'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        required=True,
        help='Path to validation CSV file'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output directory for saving the model and checkpoints'
    )
    
    # Model configuration arguments
    parser.add_argument(
        '--VSize',
        type=int,
        default=50000,
        help='Vocabulary size (default: 50000)'
    )
    
    parser.add_argument(
        '--MXPE',
        type=int,
        default=512,
        help='Maximum position embeddings (default: 512)'
    )
    
    parser.add_argument(
        '--HSize',
        type=int,
        default=768,
        help='Hidden size (default: 768)'
    )
    
    parser.add_argument(
        '--NHL',
        type=int,
        default=12,
        help='Number of hidden layers (default: 12)'
    )
    
    parser.add_argument(
        '--NAH',
        type=int,
        default=12,
        help='Number of attention heads (default: 12)'
    )
    
    parser.add_argument(
        '--ISize',
        type=int,
        default=3072,
        help='Intermediate size (default: 3072)'
    )
    
    # Training configuration arguments
    parser.add_argument(
        '--BSize',
        type=int,
        default=8,
        help='Batch size (default: 8)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs (default: 10)'
    )
    
    parser.add_argument(
        '--LR',
        type=float,
        default=2e-4,
        help='Learning rate (default: 2e-4)'
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create config from arguments
    config = ELECTRAConfig(
        vocab_size=args.VSize,
        max_position_embeddings=args.MXPE,
        hidden_size=args.HSize,
        num_hidden_layers=args.NHL,
        num_attention_heads=args.NAH,
        intermediate_size=args.ISize
    )
    
    # Initialize trainer
    trainer = XMLTrainer(
        train_file=args.train,
        val_file=args.validate,
        output_dir=args.out,
        config=config,
        batch_size=args.BSize,
        num_epochs=args.epochs,
        learning_rate=args.LR,
        checkpoint_interval=10000,  # Fixed at 10k steps
        max_checkpoints=5  # Keep only last 5 checkpoints
    )
    
    # Start training
    try:
        trained_model = trainer.train()
        print(f"Training completed successfully. Model saved to {args.out}/model")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
