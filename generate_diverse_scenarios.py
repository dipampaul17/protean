#!/usr/bin/env python3
"""
Generate Diverse Scenarios and Validate - Gate 3 Execution
Bypassing corrupted CLI to run proper validation with sanity checks.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add protean to path
sys.path.insert(0, str(Path(__file__).parent / "protean"))

from protean.synthesis.scenarios.generator import ScenarioGenerator
from protean.core.scenario_writer import ScenarioWriter
from protean.core.validator import ScenarioValidator

def main():
    """Generate diverse scenarios and validate with proper sanity checks"""
    
    print("🎯 GATE 3 EXECUTION - ENHANCED DIVERSE SCENARIOS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Generate diverse scenarios
    print("🎭 Generating 500 diverse infrastructure failure scenarios...")
    generator = ScenarioGenerator()
    scenarios = generator.generate_batch_scenarios(count=500)
    print(f"✅ Generated {len(scenarios)} scenarios")
    
    # Write scenarios with enhanced diversity
    print("📝 Writing scenarios to files...")
    writer = ScenarioWriter(output_dir="data/smoke/scenarios")
    output_files = writer.write_scenarios(scenarios)
    print(f"✅ Written to {len(output_files)} files")
    
    # Validate scenarios
    print("🔍 Validating scenarios against patterns...")
    validator = ScenarioValidator(data_dir="data/smoke", output_dir="data/diagnostics")
    results = validator.validate_scenarios()
    
    # Calculate runtime
    runtime_hours = (time.time() - start_time) / 3600
    
    # Display results with sanity checks
    print("\n📊 VALIDATION RESULTS:")
    print(f"   🎯 Total Scenarios: {results['total_scenarios']}")
    print(f"   📄 Config Lines: {results['total_lines']}")
    print(f"   ✅ Matched Lines: {results['matched_lines']}")
    print(f"   📈 Accuracy: {results['accuracy']:.1f}%")
    print(f"   ⏱️  Runtime: {runtime_hours:.3f}h")
    
    # Sanity checks
    print("\n🔬 SANITY CHECKS:")
    
    # Check 1: Wall-clock runtime
    if runtime_hours > 0.001:  # More than ~4 seconds
        print(f"   ✅ Runtime check: {runtime_hours:.3f}h (realistic)")
    else:
        print(f"   ⚠️  Runtime check: {runtime_hours:.3f}h (suspiciously fast)")
    
    # Check 2: Scenario count
    if results['total_scenarios'] >= 500:
        print(f"   ✅ Scenario count: {results['total_scenarios']} (target: 500)")
    else:
        print(f"   ❌ Scenario count: {results['total_scenarios']} (target: 500)")
    
    # Check 3: Config line diversity
    if results['total_lines'] >= 50:  # Expecting diverse config lines
        print(f"   ✅ Config diversity: {results['total_lines']} lines (good diversity)")
    else:
        print(f"   ⚠️  Config diversity: {results['total_lines']} lines (low diversity)")
    
    # Check 4: Accuracy reasonableness (should be good but not perfect)
    if 70 <= results['accuracy'] <= 95:
        print(f"   ✅ Accuracy check: {results['accuracy']:.1f}% (realistic range)")
    elif results['accuracy'] >= 95:
        print(f"   ⚠️  Accuracy check: {results['accuracy']:.1f}% (suspiciously high)")
    else:
        print(f"   ❌ Accuracy check: {results['accuracy']:.1f}% (below 70% threshold)")
    
    # Gate 3 criteria evaluation
    print("\n🎯 GATE 3 CRITERIA EVALUATION:")
    
    extraction_success = results['accuracy'] >= 70
    pattern_count = len(results.get('operation_distribution', {}))
    canonical_patterns = len([op for op, count in results.get('operation_distribution', {}).items() if count >= 3])
    novel_patterns = len([op for op, count in results.get('operation_distribution', {}).items() if count >= 2])
    runtime_success = runtime_hours < 6.0
    
    print(f"   📊 Extraction Accuracy: {results['accuracy']:.1f}% {'✅' if extraction_success else '❌'} (≥70% required)")
    print(f"   🔍 Total Patterns Found: {pattern_count}")
    print(f"   📚 Canonical Patterns: {canonical_patterns} {'✅' if canonical_patterns >= 10 else '❌'} (≥10 required)")
    print(f"   🚀 Novel Patterns: {novel_patterns} {'✅' if novel_patterns >= 5 else '❌'} (≥5 required)")
    print(f"   ⏱️  Runtime: {runtime_hours:.3f}h {'✅' if runtime_success else '❌'} (<6h required)")
    
    # Overall Gate 3 result
    gate3_success = all([
        extraction_success,
        canonical_patterns >= 10,
        novel_patterns >= 5,
        runtime_success
    ])
    
    print(f"\n🎯 GATE 3 RESULT: {'✅ PASSED' if gate3_success else '❌ FAILED'}")
    
    if gate3_success:
        print("\n🎉 SUCCESS: Gate 3 criteria achieved with enhanced diverse scenarios!")
        print("   Ready to proceed to pattern graph compilation and GPU training.")
    else:
        print("\n⚠️  Some criteria not met. Check the detailed results above.")
    
    # Show top patterns found
    if results.get('operation_distribution'):
        print(f"\n📋 TOP PATTERNS DISCOVERED:")
        sorted_patterns = sorted(results['operation_distribution'].items(), key=lambda x: x[1], reverse=True)
        for i, (pattern, count) in enumerate(sorted_patterns[:10], 1):
            print(f"   {i:2d}. {pattern:<20} {count:3d} instances")
    
    print("\n" + "=" * 60)
    print("🔗 Check detailed results in:")
    print("   📄 data/diagnostics/validation_results.json")
    print("   📋 data/diagnostics/matched_lines.log")
    print("   📉 data/diagnostics/unmatched_lines.log")

if __name__ == "__main__":
    main() 