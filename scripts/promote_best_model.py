#!/usr/bin/env python3
"""
Promote Best Model Script
Copies the validated scientific GraphSAGE model to production location
"""

import shutil
import torch
import json
from pathlib import Path
from datetime import datetime

def main():
    """Promote the scientific GraphSAGE model to production"""
    print("🚀 Promoting Best Model to Production")
    print("=" * 50)
    
    # Paths
    scientific_model_path = Path("protean/models/scientific_graphsage_embedder.pt")
    production_model_path = Path("protean/models/pattern_embedder.pt")
    
    # Check source model exists
    if not scientific_model_path.exists():
        print(f"❌ Scientific model not found at {scientific_model_path}")
        return False
    
    # Load and validate scientific model
    try:
        checkpoint = torch.load(scientific_model_path, map_location='cpu')
        metadata = checkpoint.get('training_metadata', {})
        
        print("🔬 Scientific Model Validation:")
        print(f"   Architecture: {checkpoint.get('model_architecture', 'Unknown')}")
        print(f"   Final Loss: {metadata.get('final_loss', 'N/A')}")
        print(f"   Target Achieved: {metadata.get('target_achieved', False)}")
        print(f"   Training Time: {metadata.get('training_time_hours', 0):.2f}h")
        print(f"   Triplets Used: {metadata.get('triplets_used', 0):,}")
        print(f"   Model Size: {metadata.get('model_size_mb', 0):.1f}MB")
        
        # Validation checks
        checks_passed = 0
        total_checks = 5
        
        # Check 1: Architecture
        if checkpoint.get('model_architecture') == 'GraphSAGE':
            print("   ✅ Architecture: GraphSAGE")
            checks_passed += 1
        else:
            print(f"   ❌ Architecture: {checkpoint.get('model_architecture', 'Unknown')}")
        
        # Check 2: Target achieved
        if metadata.get('target_achieved', False):
            print("   ✅ Target loss achieved")
            checks_passed += 1
        else:
            print("   ❌ Target loss not achieved")
        
        # Check 3: Training time
        training_time = metadata.get('training_time_hours', 0)
        if training_time > 0.1:  # More than 6 minutes
            print(f"   ✅ Training time: {training_time:.2f}h")
            checks_passed += 1
        else:
            print(f"   ❌ Training time too short: {training_time:.2f}h")
        
        # Check 4: Triplets
        triplets = metadata.get('triplets_used', 0)
        if triplets >= 40000:
            print(f"   ✅ Triplets: {triplets:,}")
            checks_passed += 1
        else:
            print(f"   ❌ Insufficient triplets: {triplets:,}")
        
        # Check 5: Model size
        model_size = metadata.get('model_size_mb', 0)
        if model_size < 10:  # Less than 10MB
            print(f"   ✅ Model size: {model_size:.1f}MB")
            checks_passed += 1
        else:
            print(f"   ❌ Model size too large: {model_size:.1f}MB")
        
        print(f"\n📊 Validation: {checks_passed}/{total_checks} checks passed")
        
        if checks_passed < total_checks:
            print("❌ Model validation failed. Cannot promote to production.")
            return False
        
    except Exception as e:
        print(f"❌ Error validating model: {e}")
        return False
    
    # Backup existing production model if it exists
    if production_model_path.exists():
        backup_path = production_model_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        shutil.copy2(production_model_path, backup_path)
        print(f"📦 Backed up existing model to {backup_path}")
    
    # Copy scientific model to production
    try:
        # Add promotion metadata
        promotion_metadata = {
            'promoted_at': datetime.now().isoformat(),
            'promoted_from': str(scientific_model_path),
            'promotion_validation': {
                'checks_passed': checks_passed,
                'total_checks': total_checks,
                'validation_timestamp': datetime.now().isoformat()
            }
        }
        
        # Update checkpoint with promotion info
        checkpoint['promotion_metadata'] = promotion_metadata
        
        # Save to production location
        torch.save(checkpoint, production_model_path)
        
        print(f"✅ Model promoted successfully!")
        print(f"   Source: {scientific_model_path}")
        print(f"   Target: {production_model_path}")
        print(f"   Size: {production_model_path.stat().st_size / (1024*1024):.1f}MB")
        
        # Create promotion record
        promotion_record = {
            'timestamp': datetime.now().isoformat(),
            'source_model': str(scientific_model_path),
            'target_model': str(production_model_path),
            'validation_results': {
                'checks_passed': checks_passed,
                'total_checks': total_checks,
                'architecture': checkpoint.get('model_architecture'),
                'training_metadata': metadata
            }
        }
        
        # Save promotion record
        record_path = Path("protean/models/promotion_history.json")
        if record_path.exists():
            with open(record_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(promotion_record)
        
        with open(record_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"📋 Promotion recorded in {record_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error promoting model: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 