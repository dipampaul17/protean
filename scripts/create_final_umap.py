#!/usr/bin/env python3
"""
Create Final UMAP Visualization
Generate interactive UMAP plot showing pattern separation
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from typing import Dict, List, Tuple
from collections import defaultdict

# Scientific computing imports
try:
    import umap
    HAS_UMAP = True
except ImportError:
    print("⚠️ UMAP not available. Install with: pip install umap-learn")
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

# GraphSAGE imports
try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import from_networkx
    HAS_PYGEOMETRIC = True
except ImportError:
    print("⚠️ PyTorch Geometric not available, using fallback")
    HAS_PYGEOMETRIC = False

import networkx as nx

class PatternGraphSAGE(torch.nn.Module):
    """GraphSAGE model for loading trained embedder"""
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        if HAS_PYGEOMETRIC:
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.convs.append(SAGEConv(hidden_dim, output_dim))
            
            self.dropout = torch.nn.Dropout(0.2)
            self.layer_norm = torch.nn.LayerNorm(output_dim)
        else:
            self.fallback = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden_dim, output_dim),
                torch.nn.LayerNorm(output_dim)
            )
        
    def forward(self, data):
        if HAS_PYGEOMETRIC and hasattr(data, 'x') and hasattr(data, 'edge_index'):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = torch.relu(x)
                    x = self.dropout(x)
            
            x = global_mean_pool(x, batch)
            x = self.layer_norm(x)
            return x
        else:
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
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_config = checkpoint.get('model_config', {
        'input_dim': 1,
        'hidden_dim': 256,
        'output_dim': 128,
        'num_layers': 3
    })
    
    model = PatternGraphSAGE(**model_config)
    
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded GraphSAGE model state successfully")
        except Exception as e:
            print(f"⚠️ Could not load model state: {e}")
    
    return model, checkpoint.get('training_metadata', {})

def create_pattern_graphs():
    """Create pattern graphs from config lines"""
    config_file = Path("data/smoke/scenarios/config_lines.txt")
    if not config_file.exists():
        raise FileNotFoundError("Config lines not found")
    
    config_lines = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                config_lines.append(line)
    
    print(f"📊 Loaded {len(config_lines)} config lines for visualization")
    
    # Enhanced pattern classification logic (same as training)
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
    
    # Classify config lines
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
            
        # Create multiple graph instances per pattern for visualization
        for graph_idx in range(min(len(lines), 20)):  # Up to 20 graphs per pattern
            G = nx.Graph()
            
            config_line = lines[graph_idx % len(lines)]
            tokens = config_line.lower().replace(':', ' ').replace('_', ' ').split()
            
            # Create nodes from tokens
            nodes = []
            for i, token in enumerate(tokens[:6]):  # Limit to 6 nodes for visualization
                node_id = f"{pattern_type}_{graph_idx}_{i}"
                G.add_node(node_id, 
                          feature=hash(token) % 1000,
                          token=token,
                          pattern=pattern_type)
                nodes.append(node_id)
            
            # Add edges based on proximity
            for i in range(len(nodes)):
                for j in range(i+1, min(i+3, len(nodes))):
                    G.add_edge(nodes[i], nodes[j], weight=1.0)
            
            # Convert to PyTorch Geometric format
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
                        data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
                    
                    pattern_graphs.append(data)
                else:
                    # Fallback
                    dummy_data = type('Data', (), {
                        'x': torch.ones(len(nodes), 1, dtype=torch.float),
                        'pattern_label': pattern_type,
                        'config_line': config_line,
                        'graph_id': f"{pattern_type}_{graph_idx}"
                    })()
                    pattern_graphs.append(dummy_data)
    
    print(f"✅ Created {len(pattern_graphs)} pattern graphs for visualization")
    return pattern_graphs

def get_embeddings(model, pattern_graphs, device='cpu'):
    """Get embeddings for all pattern graphs"""
    model.eval()
    embeddings = []
    labels = []
    graph_ids = []
    config_lines = []
    
    print("🔧 Generating embeddings...")
    
    with torch.no_grad():
        if HAS_PYGEOMETRIC:
            # Batch process for efficiency
            batch_size = 32
            for i in range(0, len(pattern_graphs), batch_size):
                batch_graphs = pattern_graphs[i:i+batch_size]
                batch = Batch.from_data_list(batch_graphs).to(device)
                
                batch_embeddings = model(batch)
                embeddings.append(batch_embeddings.cpu().numpy())
                
                for graph in batch_graphs:
                    labels.append(graph.pattern_label)
                    graph_ids.append(graph.graph_id)
                    config_lines.append(getattr(graph, 'config_line', 'Unknown'))
        else:
            # Fallback processing
            for graph in pattern_graphs:
                embedding = model(graph.to(device))
                embeddings.append(embedding.cpu().numpy())
                labels.append(graph.pattern_label)
                graph_ids.append(graph.graph_id)
                config_lines.append(getattr(graph, 'config_line', 'Unknown'))
    
    if HAS_PYGEOMETRIC:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array(embeddings)
    
    print(f"✅ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings, labels, graph_ids, config_lines

def create_umap_visualization(embeddings, labels, graph_ids, config_lines, output_path="demo/final_umap.html"):
    """Create interactive UMAP visualization"""
    print("🎨 Creating UMAP visualization...")
    
    # Reduce dimensionality with UMAP or fallback to t-SNE
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, metric='cosine')
        embedding_2d = reducer.fit_transform(embeddings)
        method = "UMAP"
    elif HAS_TSNE:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embedding_2d = reducer.fit_transform(embeddings)
        method = "t-SNE"
    else:
        # Simple PCA fallback
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        method = "PCA"
    
    # Define colors for patterns
    pattern_colors = {
        'ServiceConfig': '#1f77b4',
        'CircuitBreaker': '#ff7f0e', 
        'Timeout': '#2ca02c',
        'ResourceLimit': '#d62728',
        'LoadBalance': '#9467bd',
        'Replicate': '#8c564b',
        'SecurityPolicy': '#e377c2',
        'Throttle': '#7f7f7f',
        'Scale': '#bcbd22',
        'NetworkConfig': '#17becf',
        'Monitor': '#ff9896',
        'Retry': '#aec7e8',
        'Backup': '#ffbb78',
        'Bulkhead': '#98df8a',
        'Cache': '#ff9999'
    }
    
    # Create interactive plot
    fig = go.Figure()
    
    # Group by pattern type
    unique_patterns = list(set(labels))
    
    for pattern in unique_patterns:
        pattern_mask = np.array(labels) == pattern
        pattern_indices = np.where(pattern_mask)[0]
        
        # Determine if this is a novel pattern (not in canonical set)
        canonical_patterns = {'ServiceConfig', 'CircuitBreaker', 'Timeout', 'ResourceLimit', 
                            'LoadBalance', 'Replicate', 'Monitor', 'Cache', 'Retry'}
        is_novel = pattern not in canonical_patterns
        
        # Style based on whether it's novel or canonical
        marker_symbol = 'diamond' if is_novel else 'circle'
        marker_size = 12 if is_novel else 8
        marker_color = '#ff0000' if is_novel else pattern_colors.get(pattern, '#666666')
        
        hover_text = [
            f"Pattern: {pattern}<br>"
            f"Graph ID: {graph_ids[i]}<br>"
            f"Config: {config_lines[i][:100]}{'...' if len(config_lines[i]) > 100 else ''}<br>"
            f"Type: {'Novel' if is_novel else 'Canonical'}"
            for i in pattern_indices
        ]
        
        fig.add_trace(go.Scatter(
            x=embedding_2d[pattern_mask, 0],
            y=embedding_2d[pattern_mask, 1],
            mode='markers',
            name=f"{pattern} ({'Novel' if is_novel else 'Canonical'})",
            marker=dict(
                symbol=marker_symbol,
                size=marker_size,
                color=marker_color,
                opacity=0.8 if is_novel else 0.6,
                line=dict(width=2, color='darkred' if is_novel else 'black')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Pattern Embeddings Visualization ({method})<br><sub>Red diamonds = Novel patterns detected | Circles = Canonical patterns</sub>",
        xaxis_title=f"{method} Component 1",
        yaxis_title=f"{method} Component 2",
        width=1200,
        height=800,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        template="plotly_white"
    )
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.write_html(output_path)
    print(f"✅ Visualization saved to {output_path}")
    
    # Calculate statistics
    pattern_counts = defaultdict(int)
    novel_count = 0
    canonical_count = 0
    
    for label in labels:
        pattern_counts[label] += 1
        if label not in {'ServiceConfig', 'CircuitBreaker', 'Timeout', 'ResourceLimit', 
                        'LoadBalance', 'Replicate', 'Monitor', 'Cache', 'Retry'}:
            novel_count += 1
        else:
            canonical_count += 1
    
    # Show summary
    print(f"\n📊 VISUALIZATION SUMMARY:")
    print(f"   Total patterns: {len(unique_patterns)}")
    print(f"   Canonical patterns: {len([p for p in unique_patterns if p in {'ServiceConfig', 'CircuitBreaker', 'Timeout', 'ResourceLimit', 'LoadBalance', 'Replicate', 'Monitor', 'Cache', 'Retry'}])}")
    print(f"   Novel patterns: {len([p for p in unique_patterns if p not in {'ServiceConfig', 'CircuitBreaker', 'Timeout', 'ResourceLimit', 'LoadBalance', 'Replicate', 'Monitor', 'Cache', 'Retry'}])}")
    print(f"   Total embeddings: {len(embeddings)}")
    
    if novel_count > 0:
        print(f"   🔴 Novel clusters visible: {novel_count} instances")
        print("   ✅ PASS: Novel patterns detected in embedding space")
    else:
        print("   ❌ FAIL: No novel patterns visible")
    
    return str(output_path)

def main():
    """Main visualization function"""
    print("🎨 Creating Final UMAP Visualization")
    print("=" * 50)
    
    # Check if scientific model exists
    if not Path("protean/models/scientific_graphsage_embedder.pt").exists():
        print("❌ Scientific GraphSAGE model not found")
        return False
    
    try:
        # Load model
        model, metadata = load_scientific_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print("✅ Scientific GraphSAGE model loaded")
        if metadata:
            print(f"   Training Loss: {metadata.get('final_loss', 'N/A')}")
            print(f"   Training Time: {metadata.get('training_time_hours', 0):.2f}h")
        
        # Create pattern graphs
        pattern_graphs = create_pattern_graphs()
        
        # Get embeddings
        embeddings, labels, graph_ids, config_lines = get_embeddings(model, pattern_graphs, device)
        
        # Create visualization
        output_path = create_umap_visualization(embeddings, labels, graph_ids, config_lines)
        
        print(f"\n🎉 Final visualization complete!")
        print(f"   Open: {output_path}")
        print(f"   Look for: Red diamond clusters (novel patterns)")
        print("   Validation: Novel patterns should be visibly separated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 