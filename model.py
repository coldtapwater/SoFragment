# model.py
import math
import torch
import torch.nn as nn
from safetensors.torch import save_file
from typing import Optional, Dict, Any
import os
import json
from config import ELECTRAConfig

class ELECTRAEmbeddings(nn.Module):
    def __init__(self, config: ELECTRAConfig, is_generator: bool = False):
        super().__init__()
        if is_generator:
            self.hidden_size = config.generator_config["hidden_size"]
        else:
            self.hidden_size = config.hidden_size
            
        self.word_embeddings = nn.Embedding(config.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, self.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embeddings layer."""
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        # Get word embeddings
        word_embeddings = self.word_embeddings(input_ids)
        
        # Create position IDs and get position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = word_embeddings + position_embeddings
        
        # Apply layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class ELECTRAAttention(nn.Module):
    def __init__(self, config: Dict[str, int], is_generator: bool = False):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.get("attention_probs_dropout_prob", 0.1))
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Linear transformations for Query, Key, Value
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape and transpose for attention computation
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Add the mask to the attention scores
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper
        attention_probs = self.dropout(attention_probs)
        
        # Calculate context vectors
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape back to original size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer
# model.py (continued)
class ELECTRALayer(nn.Module):
    def __init__(self, config: Dict[str, int], is_generator: bool = False):
        super().__init__()
        self.attention = ELECTRAAttention(config, is_generator)
        self.intermediate = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.output = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.layernorm1 = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.layernorm2 = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config.get("hidden_dropout_prob", 0.1))
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        attention_output = self.attention(self.layernorm1(hidden_states))
        hidden_states = hidden_states + self.dropout(attention_output)

        intermediate_output = self.intermediate(self.layernorm2(hidden_states))
        intermediate_output = self.activation(intermediate_output)
        output = self.output(intermediate_output)
        output = self.dropout(output)
        output = hidden_states + output
        
        return output

class ELECTRAEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any], is_generator: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            ELECTRALayer(config, is_generator)
            for _ in range(config["num_hidden_layers"])
        ])
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class ELECTRAGenerator(nn.Module):
    def __init__(self, config: ELECTRAConfig):
        super().__init__()
        generator_config = config.generator_config
        self.embeddings = ELECTRAEmbeddings(config, is_generator=True)
        self.encoder = ELECTRAEncoder(generator_config, is_generator=True)
        
        # Adjust dimensions for generator predictions
        self.generator_predictions = nn.Linear(
            generator_config["hidden_size"],
            config.vocab_size
        )
        self.LayerNorm = nn.LayerNorm(generator_config["hidden_size"])
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        embeddings = self.embeddings(input_ids)
        hidden_states = self.encoder(embeddings, attention_mask)
        hidden_states = self.LayerNorm(hidden_states)
        prediction_scores = self.generator_predictions(hidden_states)
        return prediction_scores

class ELECTRADiscriminator(nn.Module):
    def __init__(self, config: ELECTRAConfig):
        super().__init__()
        self.embeddings = ELECTRAEmbeddings(config)
        self.encoder = ELECTRAEncoder(config.__dict__, is_generator=False)
        self.discriminator_predictions = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        embeddings = self.embeddings(input_ids)
        hidden_states = self.encoder(embeddings, attention_mask)
        logits = self.discriminator_predictions(hidden_states)
        return logits

class ELECTRA(nn.Module):
    def __init__(self, config: ELECTRAConfig):
        super().__init__()
        self.config = config
        self.generator = ELECTRAGenerator(config)
        self.discriminator = ELECTRADiscriminator(config)
        self.vocab_size = config.vocab_size
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        # Generator forward pass
        mlm_outputs = self.generator(input_ids, attention_mask)
        
        # Sample from generator predictions
        with torch.no_grad():
            probs = torch.softmax(mlm_outputs, dim=-1)
            
            # Move to CPU for sampling operation
            probs_cpu = probs.detach().cpu()
            
            # Sample on CPU
            generated_tokens = torch.multinomial(
                probs_cpu.view(-1, self.vocab_size),
                1
            )
            
            # Move back to original device
            generated_tokens = generated_tokens.to(input_ids.device)
            generated_tokens = generated_tokens.view(input_ids.shape)
            
            # Create corrupted input for discriminator
            corrupted_inputs = input_ids.clone()
            mask = (labels != -100)  # Only replace masked tokens
            corrupted_inputs[mask] = generated_tokens[mask]
        
        # Discriminator forward pass
        disc_outputs = self.discriminator(corrupted_inputs, attention_mask)
        
        # Calculate losses
        loss = None
        if labels is not None:
            gen_loss = nn.CrossEntropyLoss()(
                mlm_outputs.view(-1, self.vocab_size),
                labels.view(-1)
            )
            
            is_replaced = (corrupted_inputs != input_ids).float()
            disc_loss = nn.BCEWithLogitsLoss()(
                disc_outputs.view(-1),
                is_replaced.view(-1)
            )
            
            loss = gen_loss + 50.0 * disc_loss  # Weight factor from paper
        
        return {
            "loss": loss,
            "mlm_outputs": mlm_outputs,
            "disc_outputs": disc_outputs,
            "corrupted_inputs": corrupted_inputs
        }
    def save_pretrained(self, save_directory: str):
        """Save model state dict and config"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        # Save model weights in safetensors format
        from safetensors.torch import save_file
        
        state_dict = self.state_dict()
        tensors = {k: v.cpu() for k, v in state_dict.items()}
        save_file(tensors, os.path.join(save_directory, "model.safetensors"))

        print(f"Model saved to {save_directory}")