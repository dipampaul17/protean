#!/usr/bin/env python3
"""
Basic usage example for Protean Pattern Discovery Engine
Demonstrates pattern discovery and validation workflow.
"""

import torch
from pathlib import Path
from protean.core.validator import ScenarioValidator
from protean.synthesis.scenarios.generator import ScenarioGenerator

def main():
    """Demonstrate basic Protean usage"""
    
    print("🔧 Protean Pattern Discovery - Basic Usage Example")
    print("=" * 50)
    
    # 1. Generate synthetic scenarios
    print("\n1. Generating infrastructure scenarios...")
    generator = ScenarioGenerator()
    scenarios = generator.generate_batch_scenarios(count=10)
    print(f"   Generated {len(scenarios)} scenarios")
    
    # 2. Validate patterns
    print("\n2. Running pattern validation...")
    validator = ScenarioValidator()
    results = validator.validate_scenarios()
    
    print(f"   Accuracy: {results['accuracy']:.1f}%")
    print(f"   Patterns discovered: {len(results['operation_distribution'])}")
    print(f"   Processing time: {results['runtime_seconds']:.2f}s")
    
    # 3. Display pattern distribution
    print("\n3. Pattern Distribution:")
    for pattern, count in sorted(results['operation_distribution'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {pattern:<15}: {count:>4} instances")
    
    # 4. Load pre-trained model (if available)
    model_path = Path("models/pattern_embedder.pt")
    if model_path.exists():
        print("\n4. Loading trained model...")
        model_data = torch.load(model_path, map_location='cpu')
        print(f"   Model loaded: {len(model_data.get('vocab', {}))} vocabulary terms")
        print(f"   Pattern types: {len(model_data.get('pattern_types', []))}")
    else:
        print("\n4. No pre-trained model found - run training first")
    
    print(f"\n✅ Pattern discovery complete!")
    print(f"   Total scenarios processed: {results['total_scenarios']}")
    print(f"   Configuration lines analyzed: {results['total_lines']}")
    
    return results

if __name__ == "__main__":
    main() 