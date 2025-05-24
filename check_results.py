#!/usr/bin/env python3
"""
Check Scientific GraphSAGE Training Results
"""

import torch
import os
from pathlib import Path

def main():
    print("🔬 SCIENTIFIC GRAPHSAGE TRAINING VALIDATION")
    print("="*60)
    
    # Check if the scientific model exists
    scientific_path = "protean/models/scientific_graphsage_embedder.pt"
    old_path = "protean/models/enhanced_pattern_embedder.pt"
    
    if not os.path.exists(scientific_path):
        print(f"❌ Scientific model not found at {scientific_path}")
        return
    
    # Load and analyze the scientific model
    try:
        scientific_model = torch.load(scientific_path, map_location="cpu")
        print("✅ Scientific GraphSAGE model loaded successfully")
        
        # Check metadata
        if "training_metadata" in scientific_model:
            metadata = scientific_model["training_metadata"]
            print(f"\n📊 TRAINING RESULTS:")
            print(f"   Final Loss: {metadata.get('final_loss', 'N/A'):.4f}")
            print(f"   Target Achieved: {metadata.get('target_achieved', False)}")
            print(f"   Training Time: {metadata.get('training_time_hours', 0):.2f}h")
            print(f"   Total Epochs: {metadata.get('total_epochs', 0)}")
            print(f"   Triplets Used: {metadata.get('triplets_used', 0):,}")
            print(f"   Model Size: {metadata.get('model_size_mb', 0):.1f}MB")
            print(f"   Target Loss: {metadata.get('target_loss', 'N/A')}")
            print(f"   Min Epochs: {metadata.get('min_epochs', 'N/A')}")
        else:
            print("❌ No training metadata found")
        
        # Check architecture
        architecture = scientific_model.get("model_architecture", "Unknown")
        print(f"   Architecture: {architecture}")
        
        # Check model config
        if "model_config" in scientific_model:
            config = scientific_model["model_config"]
            print(f"\n🏗️ MODEL CONFIGURATION:")
            for key, value in config.items():
                print(f"   {key}: {value}")
    
    except Exception as e:
        print(f"❌ Error loading scientific model: {e}")
        return
    
    # Compare file sizes
    try:
        old_size = os.path.getsize(old_path) / (1024 * 1024)
        new_size = os.path.getsize(scientific_path) / (1024 * 1024)
        
        print(f"\n📊 MODEL SIZE COMPARISON:")
        print(f"   ❌ Old LSTM model: {old_size:.1f}MB")
        print(f"   ✅ New GraphSAGE model: {new_size:.1f}MB")
        print(f"   🎯 Size reduction: {((old_size - new_size) / old_size * 100):.1f}%")
        
        # Validate expectations
        print(f"\n✅ SCIENTIFIC VALIDATION:")
        if new_size < 10:
            print(f"   ✅ Model size < 10MB: {new_size:.1f}MB (Expected ~6MB)")
        else:
            print(f"   ❌ Model size too large: {new_size:.1f}MB")
            
        if architecture == "GraphSAGE":
            print(f"   ✅ Architecture is GraphSAGE (not LSTM)")
        else:
            print(f"   ❌ Architecture is not GraphSAGE: {architecture}")
            
        if "training_metadata" in scientific_model:
            training_time = metadata.get('training_time_hours', 0)
            triplets = metadata.get('triplets_used', 0)
            
            if training_time > 0.1:  # More than 6 minutes
                print(f"   ✅ Training time > 6 min: {training_time:.2f}h")
            else:
                print(f"   ❌ Training time too short: {training_time:.2f}h")
                
            if triplets > 40000:
                print(f"   ✅ Triplets > 40k: {triplets:,}")
            elif triplets > 10000:
                print(f"   ⚠️ Triplets moderate: {triplets:,} (target: >40k)")
            else:
                print(f"   ❌ Triplets insufficient: {triplets:,}")
        
    except Exception as e:
        print(f"❌ Error comparing models: {e}")

if __name__ == "__main__":
    main() 