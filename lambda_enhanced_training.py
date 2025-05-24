#!/usr/bin/env python3
"""
Enhanced Scientific GPU Training for Protean Pattern Discovery
Lambda GPU Instance Training with Comprehensive Monitoring
"""

import os
import sys
import time
import random
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import subprocess

# Scientific experiment configuration
WANDB_API_KEY = "5379ff74a6e1ccca55c7c830d999e174ff971cc3"
TARGET_LOSS = 0.40
GPU_HOURS_BUDGET = 10.0
EXPERIMENT_NAME = f"protean-enhanced-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def setup_environment():
    """Enhanced dependency setup for Lambda GPU"""
    print("🚀 Enhanced Protean Scientific GPU Training")
    print("============================================================")
    print(f"Instance: {subprocess.check_output(['hostname']).decode().strip()}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Install PyTorch with CUDA
    dependencies = [
        ("pip install --upgrade pip", "pip upgrade"),
        ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "PyTorch CUDA"),
        ("pip install wandb loguru scikit-learn scipy networkx pathlib2", "ML libraries"),
        ("pip install transformers datasets tokenizers", "NLP libraries")
    ]
    
    print("📦 Installing enhanced dependencies...")
    for cmd, desc in dependencies:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {desc} installed")
            else:
                print(f"⚠️ {desc} had issues but continuing...")
        except Exception as e:
            print(f"⚠️ {desc} installation failed: {e}")

def setup_wandb():
    """Enhanced W&B setup with proper error handling"""
    print("🎯 Setting up Weights & Biases...")
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    try:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        return True
    except ImportError:
        print("❌ wandb not available")
        return False
    except Exception as e:
        print(f"⚠️ wandb setup warning: {e}")
        return False

def verify_gpu():
    """Enhanced GPU verification with memory info"""
    print("🎮 Verifying GPU setup...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi working")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("❌ nvidia-smi failed")
            return False
            
        # Test PyTorch CUDA
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory: {memory_gb:.1f}GB")
            print(f"CUDA version: {torch.version.cuda}")
            return True
        return False
    except Exception as e:
        print(f"❌ GPU verification failed: {e}")
        return False

def load_enhanced_scenario_data():
    """Load enhanced scenario data from our validated set"""
    print("📊 Loading enhanced scenario data...")
    
    # Load config lines from our validated scenarios
    config_file = Path("data/smoke/scenarios/config_lines.txt")
    if not config_file.exists():
        raise FileNotFoundError("Enhanced scenario data not found. Run scenario generation first.")
    
    config_lines = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                config_lines.append(line)
    
    print(f"✅ Loaded {len(config_lines)} enhanced config lines")
    
    # Enhanced pattern mapping based on our validation results
    enhanced_patterns = {
        'ServiceConfig': [],
        'CircuitBreaker': [],
        'Timeout': [],
        'ResourceLimit': [],
        'LoadBalance': [],
        'Replicate': [],
        'SecurityPolicy': [],
        'Throttle': [],
        'Scale': [],
        'NetworkConfig': [],
        'Monitor': [],
        'Retry': [],
        'Backup': [],
        'Bulkhead': [],
        'Cache': []
    }
    
    # Classify config lines using our enhanced patterns
    for line in config_lines:
        line_lower = line.lower()
        
        if any(keyword in line_lower for keyword in ['service_name', 'deployment.strategy', 'scenario.id']):
            enhanced_patterns['ServiceConfig'].append(line)
        elif any(keyword in line_lower for keyword in ['circuit_breaker', 'failure_threshold', 'recovery_timeout']):
            enhanced_patterns['CircuitBreaker'].append(line)
        elif any(keyword in line_lower for keyword in ['timeout', 'connection_timeout']):
            enhanced_patterns['Timeout'].append(line)
        elif any(keyword in line_lower for keyword in ['memory_limit', 'cpu_limit', 'disk_quota']):
            enhanced_patterns['ResourceLimit'].append(line)
        elif any(keyword in line_lower for keyword in ['load_balancing', 'health_check']):
            enhanced_patterns['LoadBalance'].append(line)
        elif any(keyword in line_lower for keyword in ['replicas', 'backup_count']):
            enhanced_patterns['Replicate'].append(line)
        elif any(keyword in line_lower for keyword in ['encryption', 'auth_required', 'ssl_enabled']):
            enhanced_patterns['SecurityPolicy'].append(line)
        elif any(keyword in line_lower for keyword in ['throttle', 'rate_limit']):
            enhanced_patterns['Throttle'].append(line)
        elif any(keyword in line_lower for keyword in ['scaling', 'auto_scaling']):
            enhanced_patterns['Scale'].append(line)
        elif any(keyword in line_lower for keyword in ['network', 'proxy_config']):
            enhanced_patterns['NetworkConfig'].append(line)
        elif any(keyword in line_lower for keyword in ['monitoring', 'metrics', 'log_level']):
            enhanced_patterns['Monitor'].append(line)
        elif any(keyword in line_lower for keyword in ['retry', 'max_retries']):
            enhanced_patterns['Retry'].append(line)
        elif any(keyword in line_lower for keyword in ['backup', 'backup_schedule']):
            enhanced_patterns['Backup'].append(line)
        elif any(keyword in line_lower for keyword in ['bulkhead', 'isolation']):
            enhanced_patterns['Bulkhead'].append(line)
        elif any(keyword in line_lower for keyword in ['cache', 'cache_ttl']):
            enhanced_patterns['Cache'].append(line)
    
    # Show pattern distribution
    print("📊 Enhanced pattern distribution:")
    for pattern, lines in enhanced_patterns.items():
        if lines:
            print(f"   {pattern:<15}: {len(lines):3d} lines")
    
    return enhanced_patterns

def create_enhanced_training_instances(enhanced_patterns):
    """Create scientific training instances"""
    print("🔧 Creating enhanced training instances...")
    
    training_instances = []
    total_instances = 0
    
    for pattern, config_lines in enhanced_patterns.items():
        if not config_lines:
            continue
            
        # Create diverse instances for each pattern
        for i, config_line in enumerate(config_lines):
            training_instances.append({
                'pattern': pattern,
                'config_line': config_line,
                'confidence': random.uniform(0.75, 0.95),
                'instance_id': f"{pattern}_{i:03d}",
                'source': 'enhanced_scenarios'
            })
            total_instances += 1
    
    print(f"✅ Created {total_instances} enhanced training instances")
    return training_instances

def run_enhanced_gpu_training():
    """Run enhanced training with scientific monitoring"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    has_wandb = setup_wandb()
    if has_wandb:
        import wandb
        
        # Get GPU info for experiment metadata
        gpu_name = "Unknown"
        gpu_memory = "Unknown"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        
        wandb.init(
            project="protean-enhanced-scientific",
            name=EXPERIMENT_NAME,
            config={
                "experiment_type": "enhanced_scenario_training",
                "scientific_validation": True,
                "scenario_count": 500,
                "config_lines": 3461,
                "target_loss": TARGET_LOSS,
                "gpu_hours_budget": GPU_HOURS_BUDGET,
                "gpu_name": gpu_name,
                "gpu_memory": gpu_memory,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda
            }
        )
    
    # Load enhanced data
    enhanced_patterns = load_enhanced_scenario_data()
    training_instances = create_enhanced_training_instances(enhanced_patterns)
    
    # Build vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    for instance in training_instances:
        tokens = instance['config_line'].lower().replace(':', ' ').replace('_', ' ').split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    print(f"📚 Enhanced vocabulary: {len(vocab)} tokens")
    
    # Create triplet dataset
    triplets = []
    pattern_instances = {}
    
    # Group by pattern
    for instance in training_instances:
        pattern = instance['pattern']
        if pattern not in pattern_instances:
            pattern_instances[pattern] = []
        pattern_instances[pattern].append(instance)
    
    # Create triplets
    for pattern, instances in pattern_instances.items():
        if len(instances) < 2:
            continue
            
        for _ in range(len(instances) * 3):  # More triplets for robust training
            anchor = random.choice(instances)
            positive = random.choice([inst for inst in instances if inst != anchor])
            
            other_patterns = [p for p in pattern_instances.keys() if p != pattern]
            if other_patterns:
                negative_pattern = random.choice(other_patterns)
                negative = random.choice(pattern_instances[negative_pattern])
                
                triplets.append({
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative,
                    'anchor_pattern': pattern
                })
    
    print(f"🔗 Created {len(triplets)} enhanced triplets")
    
    # Enhanced model architecture
    class EnhancedPatternEmbedder(nn.Module):
        def __init__(self, vocab_size, embedding_dim=512, hidden_dim=1024):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.1)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True, dropout=0.1)
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        
        def forward(self, input_ids):
            embeddings = self.embedding(input_ids)
            lstm_out, _ = self.lstm(embeddings)
            attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
            pooled = attended.mean(dim=1)
            return self.projection(pooled)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedPatternEmbedder(len(vocab)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
    
    print(f"🚀 Starting enhanced GPU training on {device}")
    
    # Tokenize function
    def tokenize_line(line, vocab, max_length=64):
        tokens = ['<BOS>'] + line.lower().replace(':', ' ').replace('_', ' ').split() + ['<EOS>']
        token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([vocab['<PAD>']] * (max_length - len(token_ids)))
        
        return token_ids
    
    # Prepare training data
    anchors, positives, negatives = [], [], []
    for triplet in triplets:
        anchors.append(tokenize_line(triplet['anchor']['config_line'], vocab))
        positives.append(tokenize_line(triplet['positive']['config_line'], vocab))
        negatives.append(tokenize_line(triplet['negative']['config_line'], vocab))
    
    # Convert to tensors
    anchor_tensor = torch.tensor(anchors, dtype=torch.long)
    positive_tensor = torch.tensor(positives, dtype=torch.long)
    negative_tensor = torch.tensor(negatives, dtype=torch.long)
    
    dataset = TensorDataset(anchor_tensor, positive_tensor, negative_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop with scientific monitoring
    print("🎯 Enhanced training started...")
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (anchor_batch, positive_batch, negative_batch) in enumerate(dataloader):
            anchor_batch = anchor_batch.to(device)
            positive_batch = positive_batch.to(device)
            negative_batch = negative_batch.to(device)
            
            optimizer.zero_grad()
            
            anchor_emb = model(anchor_batch)
            positive_emb = model(positive_batch)
            negative_emb = model(negative_batch)
            
            loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        elapsed_hours = (time.time() - start_time) / 3600
        
        # Scientific logging
        if has_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "triplet_loss": avg_loss,
                "best_loss": best_loss,
                "elapsed_hours": elapsed_hours,
                "gpu_memory_used": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        if (epoch + 1) % 10 == 0 or avg_loss < TARGET_LOSS:
            print(f"Epoch {epoch+1:3d}/100: "
                  f"🎯 Loss: {avg_loss:.4f} "
                  f"⭐ Best: {best_loss:.4f} "
                  f"⏱️ {elapsed_hours:.1f}h "
                  f"🎮 GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
            
            if avg_loss < TARGET_LOSS:
                print(f"🎉 SCIENTIFIC TARGET ACHIEVED! Loss {avg_loss:.4f} < {TARGET_LOSS}")
                break
        
        # Check time budget
        if elapsed_hours > GPU_HOURS_BUDGET:
            print(f"⏰ Time budget reached: {elapsed_hours:.2f}h")
            break
    
    # Save model
    model_dir = Path("protean/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "enhanced_pattern_embedder.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'training_metadata': {
            'final_loss': best_loss,
            'target_achieved': best_loss < TARGET_LOSS,
            'training_time': elapsed_hours,
            'experiment_name': EXPERIMENT_NAME
        }
    }, model_path)
    
    training_time = elapsed_hours
    target_achieved = best_loss < TARGET_LOSS
    
    print("🎉 ENHANCED SCIENTIFIC TRAINING COMPLETED!")
    print(f"📊 Final loss: {best_loss:.4f}")
    print(f"🎯 Target achieved: {target_achieved}")
    print(f"⏱️ Training time: {training_time:.2f}h")
    print(f"💾 Model saved: {model_path}")
    
    if has_wandb:
        wandb.log({
            "final/triplet_loss": best_loss,
            "final/target_achieved": target_achieved,
            "final/training_time_hours": training_time,
            "final/model_saved": True
        })
        wandb.finish()
    
    return {
        'success': target_achieved,
        'final_loss': best_loss,
        'training_time': training_time,
        'experiment_name': EXPERIMENT_NAME
    }

def main():
    """Enhanced main execution"""
    try:
        setup_environment()
        
        if not verify_gpu():
            print("❌ GPU verification failed")
            return False
        
        results = run_enhanced_gpu_training()
        
        print("\n🔬 SCIENTIFIC EXPERIMENT RESULTS:")
        print("=" * 60)
        print(f"🎯 Target achieved: {results['success']}")
        print(f"📊 Final loss: {results['final_loss']:.4f}")
        print(f"⏱️ Training time: {results['training_time']:.2f}h")
        print(f"🧪 Experiment: {results['experiment_name']}")
        print("=" * 60)
        print("🎉 ENHANCED LAMBDA GPU TRAINING COMPLETED!")
        
        return results['success']
        
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 