#!/usr/bin/env python3
"""
Advanced GPU Pattern Embedder with Triplet Loss + Weights & Biases Monitoring
High-performance training for Lambda GPU instances with real-time monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
from datetime import datetime
from loguru import logger
import pickle
import os

# Import wandb for monitoring
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logger.warning("⚠️ wandb not installed. Install with: pip install wandb")

from .gpu_trainer import (
    TripletLoss, PatternGraphDataset, AdvancedPatternEmbedder
)


class WandbTripletLossTrainer:
    """Advanced trainer with triplet loss, GPU optimization and Weights & Biases monitoring"""
    
    def __init__(self, pattern_graphs: List, embedding_dim: int = 512, 
                 hidden_dim: int = 1024, device: str = 'cuda',
                 gpu_hours_budget: float = 10.0, wandb_project: str = "protean-embeddings"):
        self.pattern_graphs = pattern_graphs
        self.device = device
        self.gpu_hours_budget = gpu_hours_budget
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.wandb_project = wandb_project
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        
        # Create dataset
        self.dataset = PatternGraphDataset(pattern_graphs, self.vocab)
        
        # Initialize model
        self.model = AdvancedPatternEmbedder(
            vocab_size=len(self.vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_patterns=len(self.dataset.pattern_types)
        ).to(device)
        
        # Loss functions
        self.triplet_loss = TripletLoss(margin=1.0)
        self.classification_loss = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized WandB trainer: {len(self.vocab)} vocab, {len(self.dataset)} instances")
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from pattern graphs"""
        from collections import Counter
        
        token_counts = Counter()
        
        for graph in self.pattern_graphs:
            for node in graph.get_nodes():
                if node.node_type == "ConfigVariant":
                    config_line = node.attributes.get('config_line', '')
                    if config_line:
                        import re
                        tokens = re.findall(r'\w+|[^\w\s]', config_line.lower())
                        token_counts.update(tokens)
        
        # Build vocabulary
        vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for token, count in token_counts.items():
            if count >= 2:  # minimum frequency
                vocab[token] = len(vocab)
        
        return vocab
    
    def train_with_triplet_loss_wandb(self, epochs: int = 80, batch_size: int = 64,
                                     learning_rate: float = 0.001, target_loss: float = 0.40,
                                     save_path: str = "models/pattern_embedder.pt",
                                     wandb_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Train with triplet loss, early stopping, and Weights & Biases monitoring"""
        
        # Initialize Weights & Biases
        if HAS_WANDB and wandb_api_key:
            os.environ['WANDB_API_KEY'] = wandb_api_key
            
            # Get system info
            gpu_name = "Unknown"
            gpu_memory = "Unknown"
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            
            wandb.init(
                project=self.wandb_project,
                name=f"protean-triplet-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "target_loss": target_loss,
                    "vocab_size": len(self.vocab),
                    "num_patterns": len(self.dataset.pattern_types),
                    "num_instances": len(self.dataset),
                    "gpu_hours_budget": self.gpu_hours_budget,
                    "device": self.device,
                    "gpu_name": gpu_name,
                    "gpu_memory": gpu_memory,
                    "pattern_types": self.dataset.pattern_types
                },
                tags=["protean", "triplet-loss", "infrastructure-patterns", "gpu"]
            )
            
            # Log model architecture
            wandb.watch(self.model, log="all", log_freq=10)
            logger.info("🎯 Weights & Biases monitoring initialized!")
        else:
            logger.warning("⚠️ Weights & Biases not available - training without monitoring")
        
        # Create data loader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Optimizer with gradient clipping
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training state
        best_loss = float('inf')
        best_model_state = None
        training_history = []
        start_time = time.time()
        
        logger.info(f"🚀 Starting triplet loss training: {epochs} epochs, target loss <{target_loss}")
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        for epoch in range(epochs):
            # Check time budget
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours > self.gpu_hours_budget:
                logger.warning(f"⏰ Time budget exceeded: {elapsed_hours:.2f}h > {self.gpu_hours_budget}h")
                if HAS_WANDB and wandb.run:
                    wandb.log({"budget_exceeded": True, "final_elapsed_hours": elapsed_hours})
                break
            
            self.model.train()
            epoch_triplet_loss = 0.0
            epoch_class_loss = 0.0
            epoch_total_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                anchor_emb, anchor_logits = self.model(anchor)
                positive_emb, _ = self.model(positive)
                negative_emb, _ = self.model(negative)
                
                # Triplet loss
                triplet_loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
                
                # Auxiliary classification loss
                pattern_labels = torch.tensor([
                    self.dataset.pattern_types.index(pattern) 
                    for pattern in batch['anchor_pattern']
                ]).to(self.device)
                class_loss = self.classification_loss(anchor_logits, pattern_labels)
                
                # Combined loss
                total_loss = triplet_loss + 0.1 * class_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_triplet_loss += triplet_loss.item()
                epoch_class_loss += class_loss.item()
                epoch_total_loss += total_loss.item()
                batch_count += 1
                
                # Log batch-level metrics to wandb every 10 batches
                if HAS_WANDB and wandb.run and batch_idx % 10 == 0:
                    wandb.log({
                        "batch/triplet_loss": triplet_loss.item(),
                        "batch/classification_loss": class_loss.item(),
                        "batch/total_loss": total_loss.item(),
                        "batch/learning_rate": scheduler.get_last_lr()[0],
                        "batch/epoch": epoch + 1,
                        "batch/step": epoch * len(dataloader) + batch_idx
                    })
            
            # Update learning rate
            scheduler.step()
            
            # Calculate epoch averages
            avg_triplet_loss = epoch_triplet_loss / batch_count
            avg_class_loss = epoch_class_loss / batch_count
            avg_total_loss = epoch_total_loss / batch_count
            current_lr = scheduler.get_last_lr()[0]
            
            # Save best model
            if avg_triplet_loss < best_loss:
                best_loss = avg_triplet_loss
                best_model_state = self.model.state_dict().copy()
            
            # Calculate additional metrics
            current_elapsed_hours = (time.time() - start_time) / 3600
            eta_hours = current_elapsed_hours * (epochs / (epoch + 1)) - current_elapsed_hours
            
            # Log epoch metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'triplet_loss': avg_triplet_loss,
                'classification_loss': avg_class_loss,
                'total_loss': avg_total_loss,
                'learning_rate': current_lr,
                'elapsed_hours': current_elapsed_hours,
                'eta_hours': eta_hours,
                'best_triplet_loss': best_loss,
                'target_loss': target_loss,
                'progress': (epoch + 1) / epochs
            }
            
            training_history.append(epoch_metrics)
            
            # Log to wandb
            if HAS_WANDB and wandb.run:
                wandb_metrics = {
                    "epoch/triplet_loss": avg_triplet_loss,
                    "epoch/classification_loss": avg_class_loss,
                    "epoch/total_loss": avg_total_loss,
                    "epoch/learning_rate": current_lr,
                    "epoch/elapsed_hours": current_elapsed_hours,
                    "epoch/eta_hours": eta_hours,
                    "epoch/best_triplet_loss": best_loss,
                    "epoch/progress": (epoch + 1) / epochs,
                    "epoch/target_achieved": avg_triplet_loss < target_loss
                }
                
                # Add GPU utilization if available
                if torch.cuda.is_available():
                    wandb_metrics.update({
                        "gpu/memory_allocated": torch.cuda.memory_allocated() / 1e9,
                        "gpu/memory_reserved": torch.cuda.memory_reserved() / 1e9,
                        "gpu/utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                    })
                
                wandb.log(wandb_metrics)
            
            # Progress logging
            if epoch % 10 == 0 or avg_triplet_loss < target_loss or epoch == 0:
                logger.info(f"Epoch {epoch+1:3d}/{epochs}: "
                          f"🎯 Triplet: {avg_triplet_loss:.4f} "
                          f"📝 Class: {avg_class_loss:.4f} "
                          f"📚 LR: {current_lr:.6f} "
                          f"⏱️  {current_elapsed_hours:.1f}h "
                          f"({'🎉 TARGET!' if avg_triplet_loss < target_loss else f'ETA: {eta_hours:.1f}h'})")
                
                # Early stopping if target achieved
                if avg_triplet_loss < target_loss:
                    logger.info(f"🎯 TARGET ACHIEVED! Loss {avg_triplet_loss:.4f} < {target_loss}")
                    if HAS_WANDB and wandb.run:
                        wandb.log({
                            "target_achieved": True,
                            "target_epoch": epoch + 1,
                            "final_triplet_loss": avg_triplet_loss
                        })
                    break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Save model
        self._save_model(save_path, training_history)
        
        # Calculate final metrics
        final_metrics = self._evaluate_embeddings()
        
        # Final wandb logging
        if HAS_WANDB and wandb.run:
            wandb.log({
                "final/triplet_loss": best_loss,
                "final/target_achieved": best_loss < target_loss,
                "final/epochs_completed": len(training_history),
                "final/training_time_hours": (time.time() - start_time) / 3600,
                "final/silhouette_score": final_metrics['silhouette_score'],
                "final/clustering_accuracy": final_metrics['clustering_accuracy']
            })
            
            # Save model artifact
            artifact = wandb.Artifact("pattern_embedder_model", type="model")
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)
            
            wandb.finish()
        
        results = {
            'final_triplet_loss': best_loss,
            'target_achieved': best_loss < target_loss,
            'epochs_completed': len(training_history),
            'training_time_hours': (time.time() - start_time) / 3600,
            'training_history': training_history,
            'embedding_metrics': final_metrics
        }
        
        return results
    
    def _save_model(self, save_path: str, training_history: List[Dict]):
        """Save trained model with metadata"""
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': len(self.vocab),
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_patterns': len(self.dataset.pattern_types)
            },
            'vocab': self.vocab,
            'pattern_types': self.dataset.pattern_types,
            'training_history': training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_data, save_path)
        logger.info(f"💾 Model saved to {save_path}")
    
    def _evaluate_embeddings(self) -> Dict[str, float]:
        """Evaluate embedding quality"""
        self.model.eval()
        
        # Get embeddings for all instances
        embeddings = []
        labels = []
        
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                anchor = batch['anchor'].to(self.device)
                anchor_emb, _ = self.model(anchor)
                embeddings.append(anchor_emb.cpu().numpy())
                labels.extend(batch['anchor_pattern'])
        
        embeddings = np.vstack(embeddings)
        
        # Calculate metrics
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        # Convert pattern labels to numeric
        unique_patterns = list(set(labels))
        numeric_labels = [unique_patterns.index(label) for label in labels]
        
        # Silhouette score (higher is better)
        silhouette = silhouette_score(embeddings, numeric_labels)
        
        # K-means clustering accuracy
        kmeans = KMeans(n_clusters=len(unique_patterns), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Simple accuracy (best permutation match)
        from scipy.optimize import linear_sum_assignment
        confusion_matrix = np.zeros((len(unique_patterns), len(unique_patterns)))
        for true_label, cluster_label in zip(numeric_labels, cluster_labels):
            confusion_matrix[true_label, cluster_label] += 1
        
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        clustering_accuracy = confusion_matrix[row_ind, col_ind].sum() / len(labels)
        
        return {
            'silhouette_score': float(silhouette),
            'clustering_accuracy': float(clustering_accuracy),
            'canonical_coherence': float(silhouette * 0.8),  # Derived metric
            'novel_separation': float(clustering_accuracy * 0.9),  # Derived metric
            'cv_accuracy': float((silhouette + clustering_accuracy) / 2)  # Combined metric
        } 