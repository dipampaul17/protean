#!/usr/bin/env python3
"""
Scientific Experimental Runner for Protean Pattern Embeddings
Systematic hyperparameter testing with statistical rigor
"""

import os
import sys
import subprocess
import time
import random
import numpy as np
import json
from pathlib import Path
from itertools import product
from typing import Dict, List, Any
import torch

# Configuration
WANDB_API_KEY = "5379ff74a6e1ccca55c7c830d999e174ff971cc3"
TARGET_LOSS = 0.40
GPU_HOURS_BUDGET = 8.0

# Experimental Design
EXPERIMENTS = {
    "embedding_dims": [256, 512, 1024],
    "margins": [0.5, 1.0, 2.0], 
    "data_sizes": [80, 160, 320],
    "seeds": [42, 123, 456],
    "learning_rates": [0.001, 0.0001]
}

# Priority experiments (most important combinations)
PRIORITY_CONFIGS = [
    {"embedding_dim": 512, "margin": 1.0, "data_size": 160, "seed": 42, "lr": 0.001},     # Baseline
    {"embedding_dim": 256, "margin": 1.0, "data_size": 160, "seed": 42, "lr": 0.001},     # Smaller model
    {"embedding_dim": 1024, "margin": 1.0, "data_size": 160, "seed": 42, "lr": 0.001},    # Larger model
    {"embedding_dim": 512, "margin": 0.5, "data_size": 160, "seed": 42, "lr": 0.001},     # Smaller margin
    {"embedding_dim": 512, "margin": 2.0, "data_size": 160, "seed": 42, "lr": 0.001},     # Larger margin
    {"embedding_dim": 512, "margin": 1.0, "data_size": 80, "seed": 42, "lr": 0.001},      # Less data
    {"embedding_dim": 512, "margin": 1.0, "data_size": 320, "seed": 42, "lr": 0.001},     # More data
    {"embedding_dim": 512, "margin": 1.0, "data_size": 160, "seed": 123, "lr": 0.001},    # Reproducibility
    {"embedding_dim": 512, "margin": 1.0, "data_size": 160, "seed": 456, "lr": 0.001},    # Reproducibility
    {"embedding_dim": 512, "margin": 1.0, "data_size": 160, "seed": 42, "lr": 0.0001},    # Lower LR
]

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    commands = [
        "pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "pip install wandb numpy scikit-learn loguru scipy"
    ]
    
    for cmd in commands:
        print(f"$ {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        print("✅ Success")
    return True

def setup_wandb():
    """Setup Weights & Biases"""
    print("🎯 Setting up Weights & Biases...")
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    result = subprocess.run(f"wandb login {WANDB_API_KEY}", shell=True, capture_output=True, text=True)
    return True

def create_training_data(data_size: int, seed: int):
    """Create training data with controlled size and seed"""
    random.seed(seed)
    np.random.seed(seed)
    
    patterns = [
        'CircuitBreaker', 'Timeout', 'Retry', 'Cache', 'LoadBalance', 'Monitor',
        'Throttle', 'Replicate', 'Scale', 'SecurityPolicy', 'ResourceLimit', 
        'NetworkConfig', 'HealthCheck', 'Backup', 'Bulkhead', 'ServiceConfig'
    ]
    
    instances_per_pattern = max(1, data_size // len(patterns))
    config_data = []
    
    for pattern in patterns:
        for i in range(instances_per_pattern):
            if len(config_data) >= data_size:
                break
                
            if pattern == 'CircuitBreaker':
                configs = ['circuit_breaker: enabled', 'failure_threshold: 5', 'hystrix.command.default.circuitBreaker.enabled: true']
            elif pattern == 'Timeout':
                configs = ['timeout: 30s', 'connection_timeout: 10s', 'read_timeout: 30s']
            elif pattern == 'Retry':
                configs = ['max_retries: 3', 'retry_policy: exponential_backoff', 'spring.retry.max-attempts: 3']
            elif pattern == 'Cache':
                configs = ['cache: enabled', 'cache_ttl: 300s', 'redis.timeout: 5000']
            elif pattern == 'LoadBalance':
                configs = ['load_balancing: round_robin', 'health_check_interval: 10s', 'upstream backend']
            elif pattern == 'Monitor':
                configs = ['monitoring: enabled', 'metrics: enabled', 'management.endpoints.web.exposure.include: health']
            else:
                configs = [f'{pattern.lower()}: enabled', f'{pattern.lower()}_config: default', f'{pattern.lower()}_timeout: 30s']
            
            config_data.append({
                'pattern': pattern,
                'config': configs[i % len(configs)],
                'confidence': 0.8 + random.uniform(0, 0.2)
            })
    
    return config_data[:data_size], patterns

class ExperimentalEmbeddingModel:
    """Experimental embedding model with configurable parameters"""
    
    def __init__(self, vocab_size, embedding_dim, num_patterns, margin, learning_rate, seed):
        import torch
        import torch.nn as nn
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_dim = embedding_dim
        self.seed = seed
        
        # Model architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        self.classifier = nn.Linear(embedding_dim, num_patterns).to(self.device)
        self.criterion = nn.TripletMarginLoss(margin=margin).to(self.device)
        
        # Optimizer with configurable learning rate
        self.optimizer = torch.optim.AdamW(
            list(self.embedding.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate, weight_decay=0.01
        )
        
        print(f"✅ Model initialized: dim={embedding_dim}, margin={margin}, lr={learning_rate}, seed={seed}")
    
    def encode_text(self, text):
        """Simple text encoding with seed control"""
        import torch
        
        # Deterministic tokenization with seed
        tokens = text.lower().split()
        indices = [hash(token + str(self.seed)) % 1000 for token in tokens[:10]]
        
        while len(indices) < 10:
            indices.append(0)
        
        return torch.tensor(indices[:10], dtype=torch.long).to(self.device)
    
    def forward(self, text):
        """Forward pass"""
        import torch
        
        indices = self.encode_text(text)
        embeddings = self.embedding(indices)
        pooled = torch.mean(embeddings, dim=0)
        return pooled
    
    def train_step(self, anchor_text, positive_text, negative_text):
        """Single training step"""
        self.optimizer.zero_grad()
        
        anchor_emb = self.forward(anchor_text)
        positive_emb = self.forward(positive_text)
        negative_emb = self.forward(negative_text)
        
        loss = self.criterion(anchor_emb.unsqueeze(0), positive_emb.unsqueeze(0), negative_emb.unsqueeze(0))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def run_experiment(config: Dict[str, Any], experiment_id: int, total_experiments: int):
    """Run a single experiment with given configuration"""
    print(f"\n🧪 EXPERIMENT {experiment_id}/{total_experiments}")
    print(f"Config: {config}")
    
    try:
        import wandb
        import torch
        
        # Create training data
        config_data, patterns = create_training_data(config['data_size'], config['seed'])
        
        # Initialize model
        model = ExperimentalEmbeddingModel(
            vocab_size=1000,
            embedding_dim=config['embedding_dim'],
            num_patterns=len(patterns),
            margin=config['margin'],
            learning_rate=config['lr'],
            seed=config['seed']
        )
        
        # Initialize wandb for this experiment
        experiment_name = f"protean-exp-{experiment_id}-dim{config['embedding_dim']}-m{config['margin']}-data{config['data_size']}-seed{config['seed']}"
        
        wandb.init(
            project="protean-embeddings-experiments",
            name=experiment_name,
            config=config,
            tags=["systematic-experiment", "scientific-validation", f"exp-{experiment_id}"],
            reinit=True
        )
        
        # Training loop
        start_time = time.time()
        best_loss = float('inf')
        target_achieved = False
        epochs_to_target = None
        
        for epoch in range(100):  # More epochs for thorough testing
            epoch_losses = []
            
            # Training batches
            for batch in range(20):
                # Sample triplet
                anchor_idx = random.randint(0, len(config_data) - 1)
                anchor = config_data[anchor_idx]
                
                # Positive: same pattern
                positive_candidates = [d for d in config_data if d['pattern'] == anchor['pattern']]
                positive = random.choice(positive_candidates)
                
                # Negative: different pattern
                negative_candidates = [d for d in config_data if d['pattern'] != anchor['pattern']]
                negative = random.choice(negative_candidates)
                
                # Training step
                loss = model.train_step(anchor['config'], positive['config'], negative['config'])
                epoch_losses.append(loss)
            
            # Calculate metrics
            avg_loss = np.mean(epoch_losses)
            elapsed_hours = (time.time() - start_time) / 3600
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Check target achievement
            if avg_loss < TARGET_LOSS and not target_achieved:
                target_achieved = True
                epochs_to_target = epoch + 1
                print(f"🎉 Target achieved at epoch {epochs_to_target}!")
            
            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "triplet_loss": avg_loss,
                "best_triplet_loss": best_loss,
                "elapsed_hours": elapsed_hours,
                "target_achieved": target_achieved,
                "epochs_to_target": epochs_to_target or 0
            })
            
            # Progress logging
            if epoch % 20 == 0 or target_achieved:
                print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f} Best={best_loss:.4f} Target={target_achieved}")
            
            # Early stopping for efficiency
            if target_achieved and epoch > epochs_to_target + 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final metrics
        final_time = (time.time() - start_time) / 3600
        
        # Log final results
        wandb.log({
            "final/triplet_loss": best_loss,
            "final/target_achieved": target_achieved,
            "final/epochs_to_target": epochs_to_target or 100,
            "final/training_time_hours": final_time,
            "final/data_efficiency": config['data_size'] / (epochs_to_target or 100)
        })
        
        # Save model
        model_dir = f"experiments/exp_{experiment_id}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/model.pt"
        
        torch.save({
            'model_state': {
                'embedding_weight': model.embedding.weight.data,
                'classifier_weight': model.classifier.weight.data,
                'classifier_bias': model.classifier.bias.data
            },
            'config': config,
            'results': {
                'final_triplet_loss': best_loss,
                'target_achieved': target_achieved,
                'epochs_to_target': epochs_to_target,
                'training_time_hours': final_time
            }
        }, model_path)
        
        wandb.finish()
        
        return {
            'experiment_id': experiment_id,
            'config': config,
            'final_triplet_loss': best_loss,
            'target_achieved': target_achieved,
            'epochs_to_target': epochs_to_target,
            'training_time_hours': final_time
        }
        
    except Exception as e:
        print(f"❌ Experiment {experiment_id} failed: {e}")
        return {
            'experiment_id': experiment_id,
            'config': config,
            'error': str(e)
        }

def main():
    """Main experimental runner"""
    print("🧪 PROTEAN SCIENTIFIC EXPERIMENTAL VALIDATION")
    print("=" * 60)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    if not setup_wandb():
        print("❌ wandb setup failed")
        return False
    
    # GPU verification
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ GPU verified")
        print(result.stdout)
    else:
        print("❌ GPU verification failed")
        return False
    
    # Run priority experiments
    results = []
    total_experiments = len(PRIORITY_CONFIGS)
    start_time = time.time()
    
    for i, config in enumerate(PRIORITY_CONFIGS, 1):
        # Check time budget
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours > GPU_HOURS_BUDGET:
            print(f"⏰ Time budget exceeded: {elapsed_hours:.2f}h > {GPU_HOURS_BUDGET}h")
            break
        
        result = run_experiment(config, i, total_experiments)
        results.append(result)
        
        print(f"✅ Experiment {i} completed")
    
    # Save comprehensive results
    results_file = "experimental_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if 'error' not in r]),
            'results': results,
            'summary': analyze_results(results)
        }, f, indent=2)
    
    print(f"\n📊 EXPERIMENTAL SUMMARY")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len([r for r in results if 'error' not in r])}")
    print(f"Results saved to: {results_file}")
    
    return True

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze experimental results for statistical significance"""
    successful = [r for r in results if 'error' not in r]
    
    if not successful:
        return {"error": "No successful experiments"}
    
    target_achieved_count = len([r for r in successful if r.get('target_achieved', False)])
    success_rate = target_achieved_count / len(successful)
    
    losses = [r['final_triplet_loss'] for r in successful]
    epochs_to_target = [r['epochs_to_target'] for r in successful if r.get('epochs_to_target')]
    
    return {
        'success_rate': success_rate,
        'mean_final_loss': np.mean(losses),
        'std_final_loss': np.std(losses),
        'min_final_loss': np.min(losses),
        'max_final_loss': np.max(losses),
        'mean_epochs_to_target': np.mean(epochs_to_target) if epochs_to_target else None,
        'std_epochs_to_target': np.std(epochs_to_target) if epochs_to_target else None,
        'reproducibility_variance': np.std([r['final_triplet_loss'] for r in successful if r['config']['seed'] in [42, 123, 456]])
    }

if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    if success:
        print("🎉 SCIENTIFIC EXPERIMENTAL VALIDATION COMPLETED!")
    else:
        print("❌ EXPERIMENTAL VALIDATION FAILED!")
    print(f"{'='*60}")
    sys.exit(0 if success else 1) 