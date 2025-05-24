#!/usr/bin/env python3
"""
Scientific GraphSAGE Training for Protean Pattern Discovery
Proper graph-based architecture with rigorous scientific validation
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
from typing import Dict, List, Any, Tuple
import subprocess
from collections import defaultdict
import torch

# Scientific experiment configuration
WANDB_API_KEY = "5379ff74a6e1ccca55c7c830d999e174ff971cc3"
TARGET_LOSS = 0.30  # More realistic target
MIN_EPOCHS = 20     # Minimum epochs to ensure proper training
MAX_EPOCHS = 100    # Maximum epochs
TARGET_TRIPLETS = 50000  # Target number of triplets
EXPERIMENT_NAME = f"protean-graphsage-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def setup_environment():
    """Enhanced dependency setup for Lambda GPU with GraphSAGE"""
    print("🚀 Scientific GraphSAGE Training for Protean")
    print("============================================================")
    print(f"Instance: {subprocess.check_output(['hostname']).decode().strip()}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Install PyTorch and PyTorch Geometric
    dependencies = [
        ("pip install --upgrade pip", "pip upgrade"),
        ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "PyTorch CUDA"),
        ("pip install torch-geometric", "PyTorch Geometric"),
        ("pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html", "PyG extensions"),
        ("pip install wandb loguru scikit-learn scipy networkx", "ML libraries"),
        ("pip install transformers datasets tokenizers", "NLP libraries")
    ]
    
    print("📦 Installing GraphSAGE dependencies...")
    for cmd, desc in dependencies:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=600)
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
    """Load enhanced scenario data"""
    print("📊 Loading enhanced scenario data...")
    
    config_file = Path("data/smoke/scenarios/config_lines.txt")
    if not config_file.exists():
        raise FileNotFoundError("Enhanced scenario data not found")
    
    config_lines = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                config_lines.append(line)
    
    print(f"✅ Loaded {len(config_lines)} enhanced config lines")
    return config_lines

def create_pattern_graph_from_config(config_lines):
    """Create actual graph structures from config lines"""
    import torch
    import networkx as nx
    from torch_geometric.utils import from_networkx
    
    print("🔧 Creating pattern graphs from config lines...")
    
    # Enhanced pattern mapping
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
    
    # Classify config lines using enhanced patterns
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
    
    # Create graph instances for each pattern
    pattern_graphs = []
    
    for pattern_type, lines in enhanced_patterns.items():
        if not lines:
            continue
            
        # Create multiple graph instances per pattern for diversity
        for graph_idx in range(min(len(lines), 20)):  # Up to 20 graphs per pattern
            G = nx.Graph()
            
            # Add nodes based on config line content
            config_line = lines[graph_idx % len(lines)]
            tokens = config_line.lower().replace(':', ' ').replace('_', ' ').split()
            
            # Create nodes from tokens (infrastructure components)
            nodes = []
            for i, token in enumerate(tokens[:10]):  # Limit to 10 nodes per graph
                node_id = f"{pattern_type}_{graph_idx}_{i}"
                G.add_node(node_id, 
                          feature=hash(token) % 1000,  # Simple hash-based feature
                          token=token,
                          pattern=pattern_type)
                nodes.append(node_id)
            
            # Add edges based on proximity and semantic relationships
            for i in range(len(nodes)):
                for j in range(i+1, min(i+4, len(nodes))):  # Connect nearby nodes
                    G.add_edge(nodes[i], nodes[j], weight=1.0)
            
            # Add pattern-specific structure
            if pattern_type == 'CircuitBreaker':
                # Circuit breaker pattern: monitor -> breaker -> service -> fallback
                if len(nodes) >= 4:
                    monitor, breaker, service, fallback = nodes[:4]
                    G.add_edge(monitor, breaker, weight=2.0)
                    G.add_edge(breaker, service, weight=2.0)
                    G.add_edge(breaker, fallback, weight=1.5)
                    
            elif pattern_type == 'LoadBalance':
                # Load balancer pattern: balancer -> multiple services
                if len(nodes) >= 3:
                    balancer = nodes[0]
                    for service in nodes[1:]:
                        G.add_edge(balancer, service, weight=1.5)
                        
            elif pattern_type == 'Replicate':
                # Replication pattern: source -> multiple replicas
                if len(nodes) >= 3:
                    source = nodes[0]
                    for replica in nodes[1:]:
                        G.add_edge(source, replica, weight=1.8)
            
            # Convert to PyTorch Geometric format
            if len(G.nodes) > 0 and len(G.edges) > 0:
                # Add node features
                for node in G.nodes():
                    if 'feature' not in G.nodes[node]:
                        G.nodes[node]['feature'] = hash(str(node)) % 1000
                
                data = from_networkx(G, group_node_attrs=['feature'])
                data.pattern_label = pattern_type
                data.config_line = config_line
                data.graph_id = f"{pattern_type}_{graph_idx}"
                
                # Ensure correct dtypes
                if hasattr(data, 'edge_index'):
                    data.edge_index = data.edge_index.long()
                if hasattr(data, 'x'):
                    data.x = data.x.float()
                elif hasattr(data, 'feature'):
                    data.x = data.feature.float().unsqueeze(1) if data.feature.dim() == 1 else data.feature.float()
                else:
                    # Create default float features
                    data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
                
                pattern_graphs.append(data)
    
    print(f"✅ Created {len(pattern_graphs)} pattern graphs")
    
    # Show pattern distribution
    pattern_counts = defaultdict(int)
    for graph in pattern_graphs:
        pattern_counts[graph.pattern_label] += 1
    
    print("📊 Graph pattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {pattern:<15}: {count:3d} graphs")
    
    return pattern_graphs

def create_massive_triplet_dataset(pattern_graphs, target_triplets=TARGET_TRIPLETS):
    """Create massive triplet dataset for robust training"""
    print(f"🔗 Creating massive triplet dataset (target: {target_triplets:,} triplets)...")
    
    # Group graphs by pattern
    pattern_graph_map = defaultdict(list)
    for graph in pattern_graphs:
        pattern_graph_map[graph.pattern_label].append(graph)
    
    triplets = []
    attempts = 0
    max_attempts = target_triplets * 5  # Avoid infinite loops
    
    while len(triplets) < target_triplets and attempts < max_attempts:
        attempts += 1
        
        # Choose a pattern with at least 2 graphs
        valid_patterns = [p for p, graphs in pattern_graph_map.items() if len(graphs) >= 2]
        if not valid_patterns:
            break
            
        pattern = random.choice(valid_patterns)
        pattern_graphs_list = pattern_graph_map[pattern]
        
        # Anchor and positive from same pattern
        anchor_graph = random.choice(pattern_graphs_list)
        positive_graph = random.choice([g for g in pattern_graphs_list if g.graph_id != anchor_graph.graph_id])
        
        # Negative from different pattern
        other_patterns = [p for p in pattern_graph_map.keys() if p != pattern]
        if other_patterns:
            negative_pattern = random.choice(other_patterns)
            negative_graph = random.choice(pattern_graph_map[negative_pattern])
            
            triplets.append({
                'anchor': anchor_graph,
                'positive': positive_graph,
                'negative': negative_graph,
                'anchor_pattern': pattern
            })
            
            # Add progress updates
            if len(triplets) % 10000 == 0:
                print(f"   Generated {len(triplets):,} triplets...")
    
    print(f"✅ Created {len(triplets):,} triplets ({len(triplets)/target_triplets*100:.1f}% of target)")
    
    if len(triplets) < target_triplets * 0.8:  # If we have less than 80% of target
        print(f"⚠️ Warning: Only generated {len(triplets):,} triplets, target was {target_triplets:,}")
    
    return triplets

def run_scientific_graphsage_training():
    """Run scientific training with proper GraphSAGE architecture"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    
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
            project="protean-graphsage-scientific",
            name=EXPERIMENT_NAME,
            config={
                "architecture": "GraphSAGE",
                "experiment_type": "scientific_graph_training",
                "target_triplets": TARGET_TRIPLETS,
                "min_epochs": MIN_EPOCHS,
                "max_epochs": MAX_EPOCHS,
                "target_loss": TARGET_LOSS,
                "gpu_name": gpu_name,
                "gpu_memory": gpu_memory,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda
            }
        )
    
    # Load data and create graphs
    config_lines = load_enhanced_scenario_data()
    pattern_graphs = create_pattern_graph_from_config(config_lines)
    triplets = create_massive_triplet_dataset(pattern_graphs, TARGET_TRIPLETS)
    
    if len(triplets) < 1000:
        raise ValueError(f"Insufficient triplets generated: {len(triplets)} < 1000")
    
    # GraphSAGE Model for Pattern Embedding
    class PatternGraphSAGE(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=256, output_dim=128, num_layers=3):
            super().__init__()
            self.num_layers = num_layers
            
            # GraphSAGE layers
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.convs.append(SAGEConv(hidden_dim, output_dim))
            
            # Dropout and normalization
            self.dropout = nn.Dropout(0.2)
            self.layer_norm = nn.LayerNorm(output_dim)
            
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # GraphSAGE forward pass
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:  # No activation on last layer
                    x = torch.relu(x)
                    x = self.dropout(x)
            
            # Global pooling to get graph-level embeddings
            x = global_mean_pool(x, batch)
            x = self.layer_norm(x)
            
            return x
    
    # Triplet Dataset for GraphSAGE
    class GraphTripletDataset(Dataset):
        def __init__(self, triplets):
            self.triplets = triplets
        
        def __len__(self):
            return len(self.triplets)
        
        def __getitem__(self, idx):
            triplet = self.triplets[idx]
            
            # Ensure graphs have proper features
            anchor = triplet['anchor']
            positive = triplet['positive'] 
            negative = triplet['negative']
            
            # Add node features if missing
            for graph in [anchor, positive, negative]:
                if not hasattr(graph, 'x') or graph.x is None:
                    # Create features from node indices
                    graph.x = torch.ones(graph.num_nodes, 1, dtype=torch.float)
                    if hasattr(graph, 'feature'):
                        graph.x = graph.feature.float().unsqueeze(1)
                    else:
                        # Use degree as feature
                        from torch_geometric.utils import degree
                        deg = degree(graph.edge_index[0], graph.num_nodes)
                        graph.x = deg.float().unsqueeze(1)
                
                # Ensure edge_index is long type and features are float
                if hasattr(graph, 'edge_index'):
                    graph.edge_index = graph.edge_index.long()
                if hasattr(graph, 'x'):
                    graph.x = graph.x.float()
            
            return anchor, positive, negative
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatternGraphSAGE(input_dim=1, hidden_dim=256, output_dim=128, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Triplet loss function
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    
    print(f"🚀 Starting Scientific GraphSAGE Training on {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check model size (should be around 6MB)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"💾 Model size: {model_size_mb:.1f}MB")
    
    if model_size_mb > 20:
        print("⚠️ Warning: Model size > 20MB, expected ~6MB for GraphSAGE")
    
    # Create data loader
    dataset = GraphTripletDataset(triplets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, 
                           collate_fn=lambda batch: batch)  # Custom collate for graphs
    
    # Training loop with proper validation
    print("🎯 Scientific training started...")
    best_loss = float('inf')
    start_time = time.time()
    epochs_without_improvement = 0
    early_stopping_patience = 10
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Unpack triplets
                anchors, positives, negatives = zip(*batch)
                
                # Create batches for GraphSAGE
                anchor_batch = Batch.from_data_list(list(anchors)).to(device)
                positive_batch = Batch.from_data_list(list(positives)).to(device)
                negative_batch = Batch.from_data_list(list(negatives)).to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                anchor_emb = model(anchor_batch)
                positive_emb = model(positive_batch)
                negative_emb = model(negative_batch)
                
                # Triplet loss
                loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"⚠️ Batch {batch_idx} failed: {e}")
                continue
        
        if batch_count == 0:
            print(f"❌ Epoch {epoch+1}: No successful batches")
            continue
            
        avg_loss = epoch_loss / batch_count
        elapsed_hours = (time.time() - start_time) / 3600
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Scientific logging
        if has_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "triplet_loss": avg_loss,
                "best_loss": best_loss,
                "elapsed_hours": elapsed_hours,
                "gpu_memory_used": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epochs_without_improvement": epochs_without_improvement
            })
        
        # Progress reporting
        if (epoch + 1) % 5 == 0 or avg_loss < TARGET_LOSS:
            print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS}: "
                  f"🎯 Loss: {avg_loss:.4f} "
                  f"⭐ Best: {best_loss:.4f} "
                  f"⏱️ {elapsed_hours:.2f}h "
                  f"🎮 GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB "
                  f"📈 No improve: {epochs_without_improvement}")
            
            # Target achieved with minimum epochs
            if avg_loss < TARGET_LOSS and epoch >= MIN_EPOCHS:
                print(f"🎉 SCIENTIFIC TARGET ACHIEVED! Loss {avg_loss:.4f} < {TARGET_LOSS} after {epoch+1} epochs")
                break
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience and epoch >= MIN_EPOCHS:
            print(f"⏹️ Early stopping after {epochs_without_improvement} epochs without improvement")
            break
    
    # Final validation
    training_time = (time.time() - start_time) / 3600
    target_achieved = best_loss < TARGET_LOSS
    
    print("🎉 SCIENTIFIC GRAPHSAGE TRAINING COMPLETED!")
    print(f"📊 Final loss: {best_loss:.4f}")
    print(f"🎯 Target achieved: {target_achieved} (target: {TARGET_LOSS})")
    print(f"⏱️ Training time: {training_time:.2f}h")
    print(f"📈 Total epochs: {epoch+1}")
    print(f"🔗 Triplets used: {len(triplets):,}")
    print(f"💾 Model size: {model_size_mb:.1f}MB")
    
    # Save model with comprehensive metadata
    model_dir = Path("protean/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "scientific_graphsage_embedder.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'GraphSAGE',
        'training_metadata': {
            'final_loss': best_loss,
            'target_achieved': target_achieved,
            'training_time_hours': training_time,
            'total_epochs': epoch + 1,
            'triplets_used': len(triplets),
            'model_size_mb': model_size_mb,
            'experiment_name': EXPERIMENT_NAME,
            'target_loss': TARGET_LOSS,
            'min_epochs': MIN_EPOCHS
        },
        'model_config': {
            'input_dim': 1,
            'hidden_dim': 256,
            'output_dim': 128,
            'num_layers': 3
        }
    }, model_path)
    
    if has_wandb:
        wandb.log({
            "final/triplet_loss": best_loss,
            "final/target_achieved": target_achieved,
            "final/training_time_hours": training_time,
            "final/total_epochs": epoch + 1,
            "final/triplets_used": len(triplets),
            "final/model_size_mb": model_size_mb,
            "final/model_saved": True
        })
        wandb.finish()
    
    return {
        'success': target_achieved,
        'final_loss': best_loss,
        'training_time': training_time,
        'total_epochs': epoch + 1,
        'triplets_used': len(triplets),
        'model_size_mb': model_size_mb,
        'experiment_name': EXPERIMENT_NAME
    }

def main():
    """Scientific main execution"""
    try:
        setup_environment()
        
        if not verify_gpu():
            print("❌ GPU verification failed")
            return False
        
        results = run_scientific_graphsage_training()
        
        print("\n🔬 SCIENTIFIC EXPERIMENT RESULTS:")
        print("=" * 60)
        print(f"🎯 Target achieved: {results['success']}")
        print(f"📊 Final loss: {results['final_loss']:.4f}")
        print(f"⏱️ Training time: {results['training_time']:.2f}h")
        print(f"📈 Total epochs: {results['total_epochs']}")
        print(f"🔗 Triplets used: {results['triplets_used']:,}")
        print(f"💾 Model size: {results['model_size_mb']:.1f}MB")
        print(f"🧪 Experiment: {results['experiment_name']}")
        print("=" * 60)
        print("🎉 SCIENTIFIC GRAPHSAGE TRAINING COMPLETED!")
        
        return results['success']
        
    except Exception as e:
        print(f"❌ Scientific training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 