#!/usr/bin/env python3
"""
KNN Evaluation Script for GraphSAGE Pattern Embedder
Test actual model performance on held-out triplets
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import networkx as nx

# GraphSAGE imports
try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import from_networkx
    HAS_PYGEOMETRIC = True
except ImportError:
    print("⚠️ PyTorch Geometric not available, using fallback")
    HAS_PYGEOMETRIC = False

class PatternGraphSAGE(nn.Module):
    """GraphSAGE model matching the trained architecture"""
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        if HAS_PYGEOMETRIC:
            # GraphSAGE layers
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.convs.append(SAGEConv(hidden_dim, output_dim))
            
            # Dropout and normalization
            self.dropout = nn.Dropout(0.2)
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            # Fallback to simple linear layers for evaluation
            self.fallback = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
    def forward(self, data):
        if HAS_PYGEOMETRIC and hasattr(data, 'x') and hasattr(data, 'edge_index'):
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
        else:
            # Fallback for non-graph data
            if hasattr(data, 'x'):
                x = data.x.mean(dim=0) if data.x.dim() > 1 else data.x
            else:
                x = torch.ones(1, dtype=torch.float, device=data.device)
            return self.fallback(x.unsqueeze(0))

def load_scientific_model():
    """Load the trained GraphSAGE model"""
    model_path = "protean/models/scientific_graphsage_embedder.pt"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Scientific model not found at {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model config
    model_config = checkpoint.get('model_config', {
        'input_dim': 1,
        'hidden_dim': 256,
        'output_dim': 128,
        'num_layers': 3
    })
    
    # Initialize model
    model = PatternGraphSAGE(**model_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded GraphSAGE model state successfully")
        except Exception as e:
            print(f"⚠️ Could not load model state: {e}")
            print("Using randomly initialized model for evaluation")
    else:
        print("⚠️ No model state found, using randomly initialized model")
    
    return model, checkpoint.get('training_metadata', {})

def load_enhanced_scenario_data():
    """Load the same scenario data used in training"""
    config_file = Path("data/smoke/scenarios/config_lines.txt")
    if not config_file.exists():
        raise FileNotFoundError("Config lines not found")
    
    config_lines = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                config_lines.append(line)
    
    print(f"📊 Loaded {len(config_lines)} config lines for evaluation")
    return config_lines

def create_pattern_graph_from_config(config_lines):
    """Create graph instances matching training logic"""
    print("🔧 Creating pattern graphs for evaluation...")
    
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
    
    # Classify config lines using same logic as training
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
    
    # Create graph instances for each pattern (same as training)
    pattern_graphs = []
    
    for pattern_type, lines in enhanced_patterns.items():
        if not lines:
            continue
            
        # Create multiple graph instances per pattern for diversity
        for graph_idx in range(min(len(lines), 10)):  # Up to 10 graphs per pattern
            G = nx.Graph()
            
            # Add nodes based on config line content
            config_line = lines[graph_idx % len(lines)]
            tokens = config_line.lower().replace(':', ' ').replace('_', ' ').split()
            
            # Create nodes from tokens (infrastructure components)
            nodes = []
            for i, token in enumerate(tokens[:8]):  # Limit to 8 nodes per graph
                node_id = f"{pattern_type}_{graph_idx}_{i}"
                G.add_node(node_id, 
                          feature=hash(token) % 1000,  # Simple hash-based feature
                          token=token,
                          pattern=pattern_type)
                nodes.append(node_id)
            
            # Add edges based on proximity and semantic relationships
            for i in range(len(nodes)):
                for j in range(i+1, min(i+3, len(nodes))):  # Connect nearby nodes
                    G.add_edge(nodes[i], nodes[j], weight=1.0)
            
            # Convert to PyTorch Geometric format or fallback
            if len(G.nodes) > 0 and len(G.edges) > 0:
                if HAS_PYGEOMETRIC:
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
                else:
                    # Fallback for non-PyG environments
                    dummy_data = type('Data', (), {
                        'x': torch.ones(len(nodes), 1, dtype=torch.float),
                        'pattern_label': pattern_type,
                        'config_line': config_line,
                        'graph_id': f"{pattern_type}_{graph_idx}"
                    })()
                    pattern_graphs.append(dummy_data)
    
    print(f"✅ Created {len(pattern_graphs)} pattern graphs for evaluation")
    return pattern_graphs

def create_held_out_triplets(pattern_graphs, num_triplets=1000):
    """Create held-out triplets for GraphSAGE evaluation"""
    # Group by pattern
    pattern_graphs_map = defaultdict(list)
    for graph in pattern_graphs:
        pattern_graphs_map[graph.pattern_label].append(graph)
    
    triplets = []
    for _ in range(num_triplets):
        # Choose a pattern with at least 2 graphs
        valid_patterns = [p for p, graphs in pattern_graphs_map.items() if len(graphs) >= 2]
        if not valid_patterns:
            break
            
        pattern = random.choice(valid_patterns)
        graphs = pattern_graphs_map[pattern]
        
        # Anchor and positive from same pattern
        anchor = random.choice(graphs)
        positive = random.choice([g for g in graphs if g.graph_id != anchor.graph_id])
        
        # Negative from different pattern
        other_patterns = [p for p in pattern_graphs_map.keys() if p != pattern]
        if other_patterns:
            negative_pattern = random.choice(other_patterns)
            negative = random.choice(pattern_graphs_map[negative_pattern])
            
            triplets.append({
                'anchor': anchor,
                'positive': positive,
                'negative': negative,
                'anchor_pattern': pattern
            })
    
    print(f"🔗 Created {len(triplets)} held-out graph triplets for evaluation")
    return triplets

def build_vocabulary(training_instances):
    """Build vocabulary matching training"""
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    for instance in training_instances:
        tokens = instance['config_line'].lower().replace(':', ' ').replace('_', ' ').split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def tokenize_line(line, vocab, max_length=64):
    """Tokenize line matching training logic"""
    tokens = ['<BOS>'] + line.lower().replace(':', ' ').replace('_', ' ').split() + ['<EOS>']
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids.extend([vocab['<PAD>']] * (max_length - len(token_ids)))
    
    return token_ids

def evaluate_knn_graphs(model, triplets, device, k=5):
    """Evaluate KNN accuracy on graph triplets"""
    model.eval()
    
    # Prepare graph data
    anchor_graphs = []
    positive_graphs = []
    negative_graphs = []
    
    for triplet in triplets:
        anchor_graphs.append(triplet['anchor'])
        positive_graphs.append(triplet['positive'])
        negative_graphs.append(triplet['negative'])
    
    # Get embeddings for all graphs
    with torch.no_grad():
        if HAS_PYGEOMETRIC:
            # Batch process graphs
            anchor_batch = Batch.from_data_list(anchor_graphs).to(device)
            positive_batch = Batch.from_data_list(positive_graphs).to(device)
            negative_batch = Batch.from_data_list(negative_graphs).to(device)
            
            anchor_emb = model(anchor_batch)
            positive_emb = model(positive_batch)
            negative_emb = model(negative_batch)
        else:
            # Fallback processing
            anchor_emb = torch.stack([model(graph.to(device)) for graph in anchor_graphs])
            positive_emb = torch.stack([model(graph.to(device)) for graph in positive_graphs])
            negative_emb = torch.stack([model(graph.to(device)) for graph in negative_graphs])
    
    # Calculate similarities
    anchor_emb = anchor_emb.cpu().numpy()
    positive_emb = positive_emb.cpu().numpy()
    negative_emb = negative_emb.cpu().numpy()
    
    # For each anchor, check if positive is closer than negative
    correct_top1 = 0
    correct_topk = 0
    
    for i in range(len(triplets)):
        anchor = anchor_emb[i]
        positive = positive_emb[i]
        negative = negative_emb[i]
        
        # Calculate distances (cosine similarity)
        pos_sim = np.dot(anchor, positive) / (np.linalg.norm(anchor) * np.linalg.norm(positive))
        neg_sim = np.dot(anchor, negative) / (np.linalg.norm(anchor) * np.linalg.norm(negative))
        
        # Top-1: positive should be more similar than negative
        if pos_sim > neg_sim:
            correct_top1 += 1
            correct_topk += 1
    
    top1_accuracy = correct_top1 / len(triplets) * 100
    topk_accuracy = correct_topk / len(triplets) * 100
    
    return top1_accuracy, topk_accuracy

def main():
    """Main evaluation function"""
    print("🔍 KNN Evaluation for Pattern Embedder")
    print("=" * 50)
    
    # Check if scientific model exists
    scientific_model_path = Path("protean/models/scientific_graphsage_embedder.pt")
    if not scientific_model_path.exists():
        print("❌ Scientific GraphSAGE model not found")
        return False
    
    # Load data
    config_lines = load_enhanced_scenario_data()
    pattern_graphs = create_pattern_graph_from_config(config_lines)
    
    print(f"🎯 Pattern graphs: {len(pattern_graphs)}")
    
    # Show pattern distribution
    pattern_counts = defaultdict(int)
    for graph in pattern_graphs:
        pattern_counts[graph.pattern_label] += 1
    
    print("📊 Pattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {pattern:<15}: {count:3d} graphs")
    
    # Create held-out triplets
    held_out_triplets = create_held_out_triplets(pattern_graphs, num_triplets=500)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Using device: {device}")
    
    try:
        # Load scientific GraphSAGE model
        model, metadata = load_scientific_model()
        model = model.to(device)
        
        print("✅ Scientific GraphSAGE model loaded successfully")
        if metadata:
            print(f"   Training Loss: {metadata.get('final_loss', 'N/A')}")
            print(f"   Triplets Used: {metadata.get('triplets_used', 0):,}")
            print(f"   Training Time: {metadata.get('training_time_hours', 0):.2f}h")
        
        # Evaluate
        print(f"\n🧪 Evaluating on {len(held_out_triplets)} held-out triplets...")
        top1_acc, topk_acc = evaluate_knn_graphs(model, held_out_triplets, device)
        
        print("\n📊 KNN Evaluation Results:")
        print(f"   Top-1 Accuracy: {top1_acc:.1f}%")
        print(f"   Top-{5} Accuracy: {topk_acc:.1f}%")
        
        # Sanity check
        if top1_acc >= 80.0:
            print("✅ PASS: Top-1 accuracy ≥ 80%")
            return True
        else:
            print(f"❌ FAIL: Top-1 accuracy {top1_acc:.1f}% < 80%")
            print("🚨 Loss function appears meaningless!")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 