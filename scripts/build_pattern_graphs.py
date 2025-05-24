#!/usr/bin/env python3
"""
Build Pattern Graphs for Embedding Training
Compiles discovered patterns into graph structures from validation results.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import uuid
from collections import defaultdict
from loguru import logger
import sys
import os

# Add protean to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protean.protean.grammar.primitives.graph import PatternGraph, Node, Edge


@dataclass
class PatternInstance:
    """A pattern instance discovered during validation"""
    config_line: str
    pattern_type: str
    confidence: float
    source: str
    context: Dict[str, Any]


def load_discovered_patterns() -> List[PatternInstance]:
    """Load patterns from validation results"""
    patterns = []
    
    # Load from matched lines (internal validation)
    matched_file = Path("data/diagnostics/matched_lines.log")
    if matched_file.exists():
        logger.info(f"📚 Loading patterns from {matched_file}")
        with open(matched_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split(' | ', 1)
                if len(parts) == 2:
                    pattern_type = parts[0].strip()
                    config_line = parts[1].strip()
                    
                    patterns.append(PatternInstance(
                        config_line=config_line,
                        pattern_type=pattern_type,
                        confidence=0.9,
                        source='internal',
                        context={'line_number': line_num, 'file': 'matched_lines.log'}
                    ))
    
    # Load from external matched lines (real world validation)
    external_file = Path("data/diagnostics/external_matched_lines.log")
    if external_file.exists():
        logger.info(f"📚 Loading patterns from {external_file}")
        with open(external_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split(' | ', 1)
                if len(parts) == 2:
                    pattern_type = parts[0].strip()
                    config_line = parts[1].strip()
                    
                    # Extract confidence from correctness indicator
                    confidence = 0.8 if '✓' in line else 0.6
                    
                    patterns.append(PatternInstance(
                        config_line=config_line,
                        pattern_type=pattern_type,
                        confidence=confidence,
                        source='external',
                        context={'line_number': line_num, 'file': 'external_matched_lines.log'}
                    ))
    
    logger.info(f"✅ Loaded {len(patterns)} pattern instances")
    return patterns


def create_pattern_graph(pattern_type: str, instances: List[PatternInstance]) -> PatternGraph:
    """Create a pattern graph from instances of the same type"""
    graph = PatternGraph(graph_id=f"pattern_{pattern_type.lower()}")
    
    # Set metadata
    graph.metadata = {
        'pattern_type': pattern_type,
        'instance_count': len(instances),
        'sources': list(set(inst.source for inst in instances)),
        'avg_confidence': sum(inst.confidence for inst in instances) / len(instances),
        'config_complexity': {
            'min_tokens': min(len(inst.config_line.split()) for inst in instances),
            'max_tokens': max(len(inst.config_line.split()) for inst in instances),
            'avg_tokens': sum(len(inst.config_line.split()) for inst in instances) / len(instances)
        }
    }
    
    # Create core pattern node
    core_node = Node(
        node_id=f"{pattern_type}_core",
        node_type="PatternCore",
        attributes={
            'pattern_name': pattern_type,
            'frequency': len(instances),
            'confidence_score': graph.metadata['avg_confidence']
        },
        cost=1.0
    )
    graph.add_node(core_node)
    
    # Create configuration variant nodes
    config_variants = {}
    for i, instance in enumerate(instances[:5]):  # Limit to top 5 variants
        variant_id = f"{pattern_type}_variant_{i+1}"
        variant_node = Node(
            node_id=variant_id,
            node_type="ConfigVariant",
            attributes={
                'config_line': instance.config_line,
                'source': instance.source,
                'confidence': instance.confidence,
                'token_count': len(instance.config_line.split())
            },
            cost=0.5
        )
        graph.add_node(variant_node)
        config_variants[variant_id] = instance
        
        # Connect variant to core
        edge = Edge(
            source=variant_id,
            target=f"{pattern_type}_core",
            edge_type="implements",
            attributes={'strength': instance.confidence},
            weight=instance.confidence
        )
        graph.add_edge(edge)
    
    # Add context nodes for external patterns (real-world evidence)
    external_instances = [inst for inst in instances if inst.source == 'external']
    if external_instances:
        context_node = Node(
            node_id=f"{pattern_type}_real_world",
            node_type="RealWorldEvidence",
            attributes={
                'external_count': len(external_instances),
                'avg_confidence': sum(inst.confidence for inst in external_instances) / len(external_instances),
                'validation_source': 'kubernetes_configs'
            },
            cost=2.0  # Higher cost for real-world validation
        )
        graph.add_node(context_node)
        
        # Connect to core
        edge = Edge(
            source=f"{pattern_type}_real_world",
            target=f"{pattern_type}_core",
            edge_type="validates",
            attributes={'evidence_strength': len(external_instances)},
            weight=2.0
        )
        graph.add_edge(edge)
    
    return graph


def classify_patterns(patterns: List[PatternInstance]) -> Dict[str, List[str]]:
    """Classify patterns as canonical vs novel"""
    # Define canonical infrastructure patterns
    canonical_patterns = {
        'Timeout', 'Retry', 'CircuitBreaker', 'LoadBalance', 'Cache', 
        'Monitor', 'Replicate', 'Scale', 'Throttle', 'Restart'
    }
    
    pattern_types = set(p.pattern_type for p in patterns)
    
    canonical = [p for p in pattern_types if p in canonical_patterns]
    novel = [p for p in pattern_types if p not in canonical_patterns]
    
    return {
        'canonical': canonical,
        'novel': novel
    }


def build_pattern_graphs() -> str:
    """Main function to build pattern graphs"""
    logger.info("🏗️  Building pattern graphs from discovered patterns...")
    
    # Load discovered patterns
    patterns = load_discovered_patterns()
    
    if not patterns:
        raise ValueError("❌ No patterns found! Run validation first.")
    
    # Group patterns by type
    patterns_by_type = defaultdict(list)
    for pattern in patterns:
        patterns_by_type[pattern.pattern_type].append(pattern)
    
    # Classify patterns
    classification = classify_patterns(patterns)
    
    # Create pattern graphs
    pattern_graphs = []
    for pattern_type, instances in patterns_by_type.items():
        logger.info(f"🔧 Creating graph for {pattern_type} ({len(instances)} instances)")
        graph = create_pattern_graph(pattern_type, instances)
        
        # Add classification to metadata
        if pattern_type in classification['canonical']:
            graph.metadata['classification'] = 'canonical'
        else:
            graph.metadata['classification'] = 'novel'
        
        pattern_graphs.append(graph)
    
    # Save to pickle file
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pattern_graphs.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(pattern_graphs, f)
    
    # Print summary
    logger.info(f"✅ Pattern graph compilation complete!")
    logger.info(f"📊 Summary:")
    logger.info(f"   Total patterns: {len(pattern_graphs)}")
    logger.info(f"   Canonical patterns: {len(classification['canonical'])}")
    logger.info(f"   Novel patterns: {len(classification['novel'])}")
    logger.info(f"   Total instances: {len(patterns)}")
    logger.info(f"   Output file: {output_file}")
    
    print(f"\n🎯 Pattern Graph Compilation Results:")
    print(f"{'='*50}")
    print(f"Total Pattern Graphs: {len(pattern_graphs)}")
    print(f"Canonical Patterns ({len(classification['canonical'])}): {', '.join(sorted(classification['canonical']))}")
    print(f"Novel Patterns ({len(classification['novel'])}): {', '.join(sorted(classification['novel']))}")
    print(f"Total Pattern Instances: {len(patterns)}")
    print(f"Average Confidence: {sum(p.confidence for p in patterns) / len(patterns):.3f}")
    print(f"Output: {output_file}")
    
    # Show pattern details
    print(f"\n📋 Pattern Details:")
    for pattern_type in sorted(patterns_by_type.keys()):
        instances = patterns_by_type[pattern_type]
        classification_type = "📘 Canonical" if pattern_type in classification['canonical'] else "🔍 Novel"
        avg_conf = sum(inst.confidence for inst in instances) / len(instances)
        external_count = len([inst for inst in instances if inst.source == 'external'])
        print(f"   {classification_type} {pattern_type}: {len(instances)} instances (ext: {external_count}, conf: {avg_conf:.3f})")
    
    return str(output_file)


if __name__ == "__main__":
    try:
        output_file = build_pattern_graphs()
        print(f"\n🎉 Success! Pattern graphs saved to: {output_file}")
    except Exception as e:
        logger.error(f"❌ Pattern graph building failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 