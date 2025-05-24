#!/usr/bin/env python3
"""
Pattern Embedder for Protean Infrastructure Pattern Discovery
Learns vector representations of infrastructure patterns for clustering and analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
import time
from datetime import datetime


@dataclass
class PatternInstance:
    """A single pattern instance with its context"""
    config_line: str
    pattern_type: str
    confidence: float
    context: Dict[str, Any]


class PatternDataset(Dataset):
    """Dataset for pattern embedding training"""
    
    def __init__(self, pattern_instances: List[PatternInstance], vocab: Dict[str, int], max_length: int = 128):
        self.instances = pattern_instances
        self.vocab = vocab
        self.max_length = max_length
        self.pattern_to_id = {pattern: idx for idx, pattern in enumerate(set(p.pattern_type for p in pattern_instances))}
        self.id_to_pattern = {idx: pattern for pattern, idx in self.pattern_to_id.items()}
        
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        # Tokenize config line
        tokens = self._tokenize(instance.config_line)
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.vocab['<PAD>']] * (self.max_length - len(token_ids)))
        
        pattern_id = self.pattern_to_id[instance.pattern_type]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'pattern_id': torch.tensor(pattern_id, dtype=torch.long),
            'confidence': torch.tensor(instance.confidence, dtype=torch.float)
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for config lines"""
        # Split on common delimiters in config files
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return ['<BOS>'] + tokens + ['<EOS>']


class PatternEmbedder(nn.Module):
    """Neural network for learning pattern embeddings"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_patterns: int, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Attention mechanism for pattern focus
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        
        # Pattern classification head
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_patterns)
        )
        
        # Pattern embedding layer (final representations)
        self.pattern_embeddings = nn.Embedding(num_patterns, embedding_dim)
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, return_embeddings=False):
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(token_embeds)  # [batch, seq_len, hidden*2]
        
        # Self-attention over LSTM outputs
        lstm_out_transposed = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden*2]
        attn_out, attn_weights = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)  # [batch, seq_len, hidden*2]
        
        # Global max pooling for sequence representation
        seq_repr = torch.max(attn_out, dim=1)[0]  # [batch, hidden*2]
        seq_repr = self.dropout(seq_repr)
        
        # Pattern classification
        pattern_logits = self.pattern_classifier(seq_repr)
        
        # Confidence prediction
        confidence_pred = self.confidence_head(seq_repr)
        
        if return_embeddings:
            return pattern_logits, confidence_pred, seq_repr
        
        return pattern_logits, confidence_pred


class PatternEmbedderTrainer:
    """Trainer for the pattern embedder model"""
    
    def __init__(self, model: PatternEmbedder, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = []
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, learning_rate: float = 1e-3,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train the pattern embedder model"""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        criterion_classification = nn.CrossEntropyLoss()
        criterion_confidence = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        logger.info(f"🎯 Starting pattern embedder training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_class_loss = 0.0
            train_conf_loss = 0.0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                pattern_ids = batch['pattern_id'].to(self.device)
                confidences = batch['confidence'].to(self.device)
                
                optimizer.zero_grad()
                
                pattern_logits, confidence_pred = self.model(input_ids)
                
                # Classification loss
                class_loss = criterion_classification(pattern_logits, pattern_ids)
                
                # Confidence prediction loss
                conf_loss = criterion_confidence(confidence_pred.squeeze(), confidences)
                
                # Combined loss
                total_loss = class_loss + 0.1 * conf_loss  # Weight confidence loss lower
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_class_loss += class_loss.item()
                train_conf_loss += conf_loss.item()
            
            # Validation phase
            val_loss, val_metrics = self._evaluate(val_loader, criterion_classification, criterion_confidence)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            # Log progress
            avg_train_loss = train_loss / len(train_loader)
            avg_train_class = train_class_loss / len(train_loader)
            avg_train_conf = train_conf_loss / len(train_loader)
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_class_loss': avg_train_class,
                'train_conf_loss': avg_train_conf,
                'val_loss': val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_metrics['accuracy']:.3f}, "
                          f"Val F1: {val_metrics['f1']:.3f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.1f}s")
        
        # Save model if path provided
        if save_path:
            self._save_model(save_path)
        
        return {
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'final_metrics': val_metrics,
            'history': self.training_history
        }
    
    def _evaluate(self, data_loader: DataLoader, criterion_class, criterion_conf) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on validation data"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                pattern_ids = batch['pattern_id'].to(self.device)
                confidences = batch['confidence'].to(self.device)
                
                pattern_logits, confidence_pred = self.model(input_ids)
                
                class_loss = criterion_class(pattern_logits, pattern_ids)
                conf_loss = criterion_conf(confidence_pred.squeeze(), confidences)
                total_loss += (class_loss + 0.1 * conf_loss).item()
                
                # Collect predictions for metrics
                preds = torch.argmax(pattern_logits, dim=1).cpu().numpy()
                labels = pattern_ids.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, {'accuracy': accuracy, 'f1': f1}
    
    def _save_model(self, save_path: str):
        """Save the trained model"""
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': self.model.token_embedding.num_embeddings,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_patterns': self.model.pattern_classifier[-1].out_features
            },
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(save_data, save_path)
        logger.info(f"💾 Model saved to {save_path}")


def load_pattern_data(diagnostics_dir: str = "data/diagnostics") -> List[PatternInstance]:
    """Load pattern data from validation results"""
    diagnostics_path = Path(diagnostics_dir)
    
    # Load matched lines
    matched_file = diagnostics_path / "matched_lines.log"
    external_matched = diagnostics_path / "external_matched_lines.log"
    
    instances = []
    
    # Load internal validation data
    if matched_file.exists():
        with open(matched_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split(' | ', 1)
                if len(parts) == 2:
                    pattern_type = parts[0].strip()
                    config_line = parts[1].strip()
                    
                    instances.append(PatternInstance(
                        config_line=config_line,
                        pattern_type=pattern_type,
                        confidence=0.9,  # High confidence for internal validation
                        context={'source': 'internal'}
                    ))
    
    # Load external validation data (more realistic)
    if external_matched.exists():
        with open(external_matched, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split(' | ', 1)
                if len(parts) == 2:
                    pattern_type = parts[0].strip()
                    config_line = parts[1].strip()
                    
                    # Extract confidence from correctness indicator
                    confidence = 0.8 if '✓' in line else 0.6
                    
                    instances.append(PatternInstance(
                        config_line=config_line,
                        pattern_type=pattern_type,
                        confidence=confidence,
                        context={'source': 'external'}
                    ))
    
    logger.info(f"📚 Loaded {len(instances)} pattern instances for training")
    return instances


def build_vocabulary(instances: List[PatternInstance], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from pattern instances"""
    from collections import Counter
    
    token_counts = Counter()
    
    for instance in instances:
        # Simple tokenization
        import re
        tokens = re.findall(r'\w+|[^\w\s]', instance.config_line.lower())
        token_counts.update(tokens)
    
    # Build vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    
    logger.info(f"📝 Built vocabulary with {len(vocab)} tokens")
    return vocab


def train_pattern_embedder(output_dir: str = "protean/models", 
                          diagnostics_dir: str = "data/diagnostics",
                          embedding_dim: int = 256,
                          hidden_dim: int = 512,
                          num_epochs: int = 100,
                          batch_size: int = 32) -> str:
    """Main training function"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("🔧 Loading pattern data...")
    instances = load_pattern_data(diagnostics_dir)
    
    if len(instances) < 10:
        raise ValueError(f"Not enough training data: {len(instances)} instances")
    
    # Build vocabulary
    vocab = build_vocabulary(instances)
    
    # Create dataset
    dataset = PatternDataset(instances, vocab)
    
    # Train/validation split
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42, 
        stratify=[inst.pattern_type for inst in instances]
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🚀 Using device: {device}")
    
    model = PatternEmbedder(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_patterns=len(dataset.pattern_to_id)
    )
    
    # Initialize trainer
    trainer = PatternEmbedderTrainer(model, device)
    
    # Train model
    model_path = output_path / "pattern_embedder.pt"
    results = trainer.train(
        train_loader, val_loader, 
        num_epochs=num_epochs,
        save_path=str(model_path)
    )
    
    # Save additional metadata
    metadata = {
        'vocab': vocab,
        'pattern_to_id': dataset.pattern_to_id,
        'id_to_pattern': dataset.id_to_pattern,
        'model_config': {
            'vocab_size': len(vocab),
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_patterns': len(dataset.pattern_to_id)
        },
        'training_results': results,
        'num_instances': len(instances)
    }
    
    metadata_path = output_path / "pattern_embedder_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"🎯 Pattern embedder training complete!")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Metadata: {metadata_path}")
    logger.info(f"   Final validation accuracy: {results['final_metrics']['accuracy']:.3f}")
    logger.info(f"   Final validation F1: {results['final_metrics']['f1']:.3f}")
    
    return str(model_path)


if __name__ == "__main__":
    train_pattern_embedder() 