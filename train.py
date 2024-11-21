import os
import argparse
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from tqdm import tqdm
import logging
import time
from datetime import datetime
import glob
import shutil
import json
import psutil
import signal
from safetensors.torch import save_model
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

from tokenizerTrainer import TokenizerTrainer
from config import ELECTRAConfig
from model import ELECTRA

class ProcessMonitor:
    def __init__(self, status_file):
        self.status_file = status_file
        self.process = psutil.Process()
        self.start_time = time.time()
        self.throttle_start = None
        self.current_throttle_level = 0  # 0: none, 1: slight, 2: severe
        self.original_batch_size = None
        self.reduced_batch_size = False
        
    def get_system_stats(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        process_time = time.time() - self.start_time
        return cpu_percent, memory_percent, process_time
    
    def should_throttle(self, cpu_percent, memory_percent, process_time):
        # Reset throttling if we've been running cool for a while
        if self.throttle_start and time.time() - self.throttle_start > 3600:  # 1 hour
            self.throttle_start = None
            self.current_throttle_level = 0
            return 0
        
        # Check if we need to start or increase throttling
        if cpu_percent > 85 or memory_percent > 85:
            if not self.throttle_start:
                self.throttle_start = time.time()
            
            throttle_duration = time.time() - self.throttle_start
            
            if throttle_duration > 3600:  # After 1 hour of high usage
                return 2  # Severe throttling
            elif throttle_duration > 1800:  # After 30 minutes of high usage
                return 1  # Slight throttling
        
        return 0
    
    def get_throttle_delay(self, level):
        if level == 1:
            return 0.1  # 100ms delay for slight throttling
        elif level == 2:
            return 0.3  # 300ms delay for severe throttling
        return 0
    
    def update_status(self, batch_num, total_batches, dataset_position, total_dataset_size, 
                     loss=None, checkpoint_info=None):
        status = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'batch_num': batch_num,
            'total_batches': total_batches,
            'dataset_position': dataset_position,
            'total_dataset_size': total_dataset_size,
            'throttle_level': self.current_throttle_level,
            'reduced_batch_size': self.reduced_batch_size,
            'loss': float(loss) if loss is not None else None,
            'latest_checkpoint': checkpoint_info,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f)

class StreamingDataset(IterableDataset):
    def __init__(self, dataset_name, split="train", batch_size=1000):
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.batch_size = batch_size
        # Get approximate dataset size
        self.total_size = self.dataset.info.splits[split].num_examples if hasattr(self.dataset.info, 'splits') else None

    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                batch = next(iterator)
                yield batch
            except StopIteration:
                break

class Trainer:
    def __init__(
        self,
        output_dir: str,
        config: ELECTRAConfig = None,
        batch_size: int = 8,
        num_epochs: int = 10,
        learning_rate: float = 2e-4,
        checkpoint_interval: int = 10000,
        max_checkpoints: int = 5,
        continue_from_checkpoint: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.config = config or self._get_default_config()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.continue_from_checkpoint = continue_from_checkpoint
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Setup device
        self.device = self.get_device()
        
        # Initialize process monitor
        self.status_file = self.output_dir / "training_status.json"
        self.monitor = ProcessMonitor(self.status_file)
        
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
        """Train tokenizer on streaming dataset."""
        self.logger.info("Preparing tokenizer...")
        
        if (self.tokenizer_dir / "tokenizer.json").exists() and self.continue_from_checkpoint:
            self.logger.info("Loading existing tokenizer...")
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(self.tokenizer_dir / "tokenizer.json")
            )
        else:
            self.logger.info("Training new tokenizer...")
            # Create temporary file for tokenizer training
            temp_dir = self.output_dir / "temp_tokenizer_files"
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / "sample_text.txt"
            
            try:
                # Get a sample of text for tokenizer training
                dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for i, example in enumerate(dataset):
                        f.write(example['text'] + '\n')
                        if i >= 10000:  # Use first 10k examples for tokenizer training
                            break
                
                # Train tokenizer
                trainer = TokenizerTrainer(vocab_size=self.config.vocab_size)
                trainer.train([str(temp_file)], self.tokenizer_dir)
                self.tokenizer = trainer.tokenizer
                
                # Save additional tokenizer files
                self.tokenizer.save(str(self.tokenizer_dir / "tokenizer.json"))
                
                # Save vocab and merges
                with open(self.tokenizer_dir / "vocab.json", 'w') as f:
                    json.dump(self.tokenizer.get_vocab(), f)
                
                if hasattr(self.tokenizer, 'get_merges'):
                    with open(self.tokenizer_dir / "merges.txt", 'w') as f:
                        f.write('\n'.join(self.tokenizer.get_merges()))
                
                # Save tokenizer config
                with open(self.tokenizer_dir / "tokenizer_config.json", 'w') as f:
                    json.dump({
                        "max_length": self.config.max_position_embeddings,
                        "vocab_size": self.config.vocab_size,
                    }, f)
                
            finally:
                shutil.rmtree(temp_dir)
                self.logger.info("Cleaned up temporary tokenizer training files")
    
    def prepare_data(self):
        """Prepare streaming dataset."""
        self.logger.info("Preparing datasets...")
        
        self.train_dataset = StreamingDataset(
            "HuggingFaceFW/fineweb",
            split="train",
            batch_size=self.batch_size
        )
        
        self.val_dataset = StreamingDataset(
            "HuggingFaceFW/fineweb",
            split="validation",
            batch_size=self.batch_size
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,  # Already batched by StreamingDataset
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=0
        )
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint file."""
        checkpoints = sorted(glob.glob(str(self.checkpoints_dir / "checkpoint_*.pt")))
        return checkpoints[-1] if checkpoints else None
    
    def save_model_files(self, model, path):
        """Save all necessary model files."""
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        # Save model weights in safetensors format
        save_model(model, path / "model.safetensors")
        
        # Save model index
        with open(path / "model.safetensors.index.json", 'w') as f:
            json.dump({
                "metadata": {"format": "pt"},
                "weight_map": {"model": "model.safetensors"}
            }, f)
        
        # Save config
        with open(path / "config.json", 'w') as f:
            json.dump(self.config.to_dict(), f)
        
        # Save generation config
        with open(path / "generation_config.json", 'w') as f:
            json.dump({
                "max_length": self.config.max_position_embeddings,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }, f)
    
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
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Manage checkpoint history
        self.manage_checkpoints()
    
    def load_checkpoint(self, model, optimizer, scheduler):
        """Load the latest checkpoint if it exists."""
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            self.logger.info(f"Loading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            return start_epoch, start_step
        return 0, 0
    
    def validate(self, model, val_loader):
        """Run validation and return average loss."""
        model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 100:  # Limit validation to 100 batches to save time
                    break
                inputs = self.tokenizer(
                    batch['text'],
                    max_length=self.config.max_position_embeddings,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        model.train()
        return avg_loss
    
    def train(self):
        """Main training loop with system monitoring and throttling."""
        self.logger.info("Starting training preparation...")
        
        # Prepare tokenizer and data
        self.prepare_tokenizer()
        self.prepare_data()
        
        # Initialize model
        model = ELECTRA(self.config)
        model.to(self.device)
        
        # Setup optimization
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        # Load checkpoint if continuing training
        start_epoch = 0
        step = 0
        if self.continue_from_checkpoint:
            start_epoch, step = self.load_checkpoint(model, optimizer, scheduler)
        
        # Store original batch size for potential dynamic adjustment
        self.monitor.original_batch_size = self.batch_size
        
        # Training loop
        self.logger.info("Starting training...")
        best_val_loss = float('inf')
        training_start_time = time.time()
        
        try:
            for epoch in range(start_epoch, self.num_epochs):
                epoch_start_time = time.time()
                model.train()
                epoch_loss = 0
                num_batches = 0
                
                progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    try:
                        # Monitor system resources and apply throttling
                        cpu_percent, memory_percent, process_time = self.monitor.get_system_stats()
                        throttle_level = self.monitor.should_throttle(cpu_percent, memory_percent, process_time)
                        
                        # Apply throttling delay if needed
                        if throttle_level > 0:
                            time.sleep(self.monitor.get_throttle_delay(throttle_level))
                        
                        # Reduce batch size if memory is too high
                        if memory_percent > 85 and not self.monitor.reduced_batch_size:
                            self.batch_size = self.batch_size // 2
                            self.monitor.reduced_batch_size = True
                            self.logger.info(f"Reduced batch size to {self.batch_size}")
                        
                        # Process batch
                        inputs = self.tokenizer(
                            batch['text'],
                            max_length=self.config.max_position_embeddings,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        # Move to device
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Forward pass
                        outputs = model(**inputs)
                        loss = outputs["loss"]
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        # Update tracking
                        epoch_loss += loss.item()
                        step += 1
                        num_batches += 1
                        
                        # Update status
                        dataset_position = batch_idx * self.batch_size
                        total_dataset_size = self.train_dataset.total_size or 0
                        checkpoint_info = None
                        
                        # Save checkpoint every 10k steps
                        if step % self.checkpoint_interval == 0:
                            self.save_checkpoint(model, optimizer, scheduler, epoch, step, loss.item())
                            checkpoint_info = f"Checkpoint saved at step {step}"
                            # Also save complete model files periodically
                            self.save_model_files(model, self.model_dir / f"checkpoint_{step}")
                        
                        # Update status file
                        self.monitor.update_status(
                            batch_num=batch_idx,
                            total_batches=len(self.train_loader),
                            dataset_position=dataset_position,
                            total_dataset_size=total_dataset_size,
                            loss=loss.item(),
                            checkpoint_info=checkpoint_info
                        )
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{epoch_loss/num_batches:.4f}',
                            'throttle': throttle_level
                        })
                    
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
                    self.save_model_files(model, self.model_dir / "best_model")
                    self.logger.info("Saved best model")
                
                scheduler.step()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted. Saving checkpoint...")
            self.save_checkpoint(model, optimizer, scheduler, epoch, step, loss.item())
            self.save_model_files(model, self.model_dir / "interrupted_model")
            raise
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            self.save_checkpoint(model, optimizer, scheduler, epoch, step, loss.item())
            self.save_model_files(model, self.model_dir / "error_model")
            raise
        
        # Save final model
        self.save_model_files(model, self.model_dir / "final_model")
        self.logger.info(f"Saved final model to {self.model_dir / 'final_model'}")
        
        total_training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_training_time/3600:.2f} hours")
        
        return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train ELECTRA model on fineweb dataset')
    
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output directory for saving the model and checkpoints'
    )
    
    parser.add_argument(
        '--continue',
        action='store_true',
        help='Continue training from latest checkpoint'
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
    trainer = Trainer(
        output_dir=args.out,
        config=config,
        batch_size=args.BSize,
        num_epochs=args.epochs,
        learning_rate=args.LR,
        checkpoint_interval=10000,  # Fixed at 10k steps
        max_checkpoints=5,  # Keep only last 5 checkpoints
        continue_from_checkpoint=getattr(args, 'continue', False)
    )
    
    # Start training
    try:
        trained_model = trainer.train()
        print(f"Training completed successfully. Model saved to {args.out}/model")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Latest state has been saved.")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
