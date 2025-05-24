#!/usr/bin/env python3
"""
Protean CLI - Infrastructure Pattern Discovery Engine
"""

import click
from pathlib import Path


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """
    Protean - Infrastructure Pattern Discovery Engine
    
    Build a pattern extraction pipeline that:
    - Recognizes 10+ canonical patterns with 70%+ accuracy
    - Discovers 5-10 genuinely unknown patterns (novelty signal)
    - Creates embeddings that cluster patterns meaningfully
    - Completes full pipeline in <6 CPU-hours (efficiency matters)
    """
    pass


@main.command()
@click.option("--output-dir", "-o", type=click.Path(), default="data/synthetic",
              help="Output directory for generated data")
@click.option("--count", "-c", type=int, default=1000,
              help="Number of synthetic patterns to generate")
@click.option("--max-time", type=int, default=7200,
              help="Maximum time in seconds (default: 2 hours)")
def generate(output_dir: str, count: int, max_time: int) -> None:
    """Generate synthetic infrastructure patterns."""
    click.echo(f"🔧 Generating {count} synthetic patterns")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"⏱️  Max time: {max_time}s (2 CPU-hours)")
    
    # TODO: Implement synthetic pattern generation
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo("✅ Synthetic generation complete")


@main.command("generate-scenarios")
@click.option("--count", "-c", type=int, default=50,
              help="Number of scenarios to generate")
@click.option("--output", "-o", type=click.Path(), default="data/smoke/scenarios",
              help="Output directory for generated scenarios")
@click.option("--categories", type=str, default=None,
              help="Comma-separated list of scenario categories to include")
def generate_scenarios(count: int, output: str, categories: str) -> None:
    """Generate infrastructure failure scenarios for testing."""
    click.echo(f"🎭 Generating {count} infrastructure failure scenarios")
    click.echo(f"📁 Output directory: {output}")
    
    # Import here to avoid circular imports
    from protean.synthesis.scenarios import ScenarioGenerator
    from protean.core.scenario_writer import ScenarioWriter
    
    try:
        # Initialize generator
        generator = ScenarioGenerator()
        writer = ScenarioWriter(output_dir=output)
        
        # Parse categories if provided
        target_categories = None
        if categories:
            target_categories = [cat.strip() for cat in categories.split(',')]
            click.echo(f"🎯 Targeting categories: {target_categories}")
        
        # Generate scenarios
        scenarios = generator.generate_batch_scenarios(
            count=count, 
            categories=target_categories
        )
        
        # Write scenarios to files
        output_files = writer.write_scenarios(scenarios)
        
        click.echo(f"✅ Generated {len(scenarios)} scenarios")
        click.echo(f"📝 Written to {len(output_files)} files in {output}")
        
        # Show summary
        category_counts = {}
        for scenario in scenarios:
            category_counts[scenario.category] = category_counts.get(scenario.category, 0) + 1
        
        click.echo("📊 Category distribution:")
        for category, count in sorted(category_counts.items()):
            click.echo(f"   {category}: {count} scenarios")
            
    except Exception as e:
        click.echo(f"❌ Error generating scenarios: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.option("--input-dir", "-i", type=click.Path(exists=True), default="data/synthetic",
              help="Input directory with patterns to extract from")
@click.option("--output-dir", "-o", type=click.Path(), default="data/canonical",
              help="Output directory for extracted patterns")
@click.option("--max-time", type=int, default=10800,
              help="Maximum time in seconds (default: 3 hours)")
def extract(input_dir: str, output_dir: str, max_time: int) -> None:
    """Extract and discover infrastructure patterns."""
    click.echo(f"🔍 Extracting patterns from {input_dir}")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"⏱️  Max time: {max_time}s (3 CPU-hours)")
    
    # TODO: Implement pattern extraction
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo("✅ Pattern extraction complete")


@main.command()
@click.option("--data", "-d", type=click.Path(exists=True), default="data/smoke",
              help="Data directory with scenarios to validate")
@click.option("--max-scenarios", type=int, default=50,
              help="Maximum number of scenarios to validate")
@click.option("--output-dir", "-o", type=click.Path(), default="data/diagnostics",
              help="Output directory for validation results")
@click.option("--max-time", type=int, default=3600,
              help="Maximum time in seconds (default: 1 hour)")
def validate(data: str, max_scenarios: int, output_dir: str, max_time: int) -> None:
    """Validate infrastructure failure scenarios and patterns."""
    click.echo(f"✅ Validating scenarios from {data}")
    click.echo(f"🎯 Max scenarios: {max_scenarios}")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"⏱️  Max time: {max_time}s (1 CPU-hour)")
    
    # Import here to avoid circular imports
    from protean.core.validator import ScenarioValidator
    
    try:
        # Initialize validator
        validator = ScenarioValidator(
            data_dir=data,
            output_dir=output_dir,
            max_scenarios=max_scenarios
        )
        
        # Run validation
        results = validator.validate_scenarios()
        
        # Report results
        total_scenarios = results['total_scenarios']
        accuracy = results['accuracy']
        matched_lines = results['matched_lines']
        total_lines = results['total_lines']
        
        click.echo(f"📊 Validation Results:")
        click.echo(f"   Scenarios processed: {total_scenarios}")
        click.echo(f"   Lines matched: {matched_lines}/{total_lines}")
        click.echo(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy < 50.0:
            click.echo("⚠️  Accuracy below 50% threshold!")
            click.echo(f"📝 Check unmatched lines in: {output_dir}/unmatched_lines.log")
            click.echo("💡 Consider adding missing regex patterns or enabling GPT fallback")
        else:
            click.echo("✅ Validation passed!")
            
        # Show diagnostics path
        click.echo(f"📋 Full diagnostics available in: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error during validation: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.option("--patterns-dir", "-p", type=click.Path(exists=True), default="data/canonical",
              help="Directory with patterns to replay")
@click.option("--scenario", "-s", type=str, required=True,
              help="Scenario name to replay")
def replay(patterns_dir: str, scenario: str) -> None:
    """Replay infrastructure patterns in scenarios."""
    click.echo(f"🎬 Replaying scenario '{scenario}' from {patterns_dir}")
    
    # TODO: Implement pattern replay
    
    click.echo("✅ Pattern replay complete")


@main.command()
@click.option("--data-dir", "-d", type=click.Path(exists=True), default="data",
              help="Data directory to visualize")
@click.option("--output-dir", "-o", type=click.Path(), default="demo/visualizations",
              help="Output directory for visualizations")
def visualize(data_dir: str, output_dir: str) -> None:
    """Create visualizations of discovered patterns."""
    click.echo(f"📊 Creating visualizations from {data_dir}")
    click.echo(f"📁 Output directory: {output_dir}")
    
    # TODO: Implement pattern visualization
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo("✅ Visualization complete")


@main.command("batch-generate")
@click.option("--count", "-c", type=int, default=500,
              help="Number of scenarios to generate")
@click.option("--outdir", "-o", type=click.Path(), default="data/synthetic/scenarios",
              help="Output directory for generated scenarios")
@click.option("--max-time", type=int, default=21600,
              help="Maximum time in seconds (default: 6 hours)")
def batch_generate(count: int, outdir: str, max_time: int) -> None:
    """Generate large scenario batch & run validation immediately (Gate 3)."""
    import time
    from pathlib import Path
    
    start_time = time.time()
    click.echo(f"🎭 Starting batch generation of {count} scenarios")
    click.echo(f"📁 Output directory: {outdir}")
    click.echo(f"⏱️  Max time: {max_time}s (6 CPU-hours)")
    click.echo("🎯 Target: Gate 3 criteria")
    
    try:
        # Import here to avoid circular imports
        from protean.synthesis.scenarios import ScenarioGenerator
        from protean.core.scenario_writer import ScenarioWriter
        from protean.core.validator import ScenarioValidator
        
        # Phase 1: Generate scenarios
        click.echo("\n📝 Phase 1: Generating scenarios...")
        generator = ScenarioGenerator()
        writer = ScenarioWriter(output_dir=outdir)
        
        scenarios = generator.generate_batch_scenarios(count=count)
        output_files = writer.write_scenarios(scenarios)
        
        # Show generation summary
        category_counts = {}
        for scenario in scenarios:
            category_counts[scenario.category] = category_counts.get(scenario.category, 0) + 1
        
        click.echo(f"✅ Generated {len(scenarios)} scenarios")
        click.echo("📊 Category distribution:")
        for category, cnt in sorted(category_counts.items()):
            click.echo(f"   {category}: {cnt} scenarios")
        
        # Phase 2: Validate scenarios
        click.echo(f"\n🔍 Phase 2: Validating {count} scenarios...")
        
        # Use parent directory of outdir for validation data source
        data_dir = str(Path(outdir).parent)
        diagnostics_dir = "data/diagnostics"
        
        validator = ScenarioValidator(
            data_dir=data_dir,
            output_dir=diagnostics_dir,
            max_scenarios=count
        )
        
        results = validator.validate_scenarios()
        
        # Calculate runtime
        runtime_seconds = time.time() - start_time
        runtime_hours = runtime_seconds / 3600
        
        # Gate 3 criteria evaluation
        click.echo(f"\n🎯 Gate 3 Evaluation:")
        click.echo(f"{'='*50}")
        
        # Criteria 1: Extraction accuracy ≥ 0.70
        accuracy = results['accuracy']
        accuracy_pass = accuracy >= 70.0
        status_1 = "✅ PASS" if accuracy_pass else "❌ FAIL"
        click.echo(f"Extraction accuracy: {accuracy:.1f}% (≥70% required) {status_1}")
        
        # Criteria 2: Canonical patterns found ≥ 10
        # Count distinct operations from operation_distribution
        canonical_patterns = len(results.get('operation_distribution', {}))
        canonical_pass = canonical_patterns >= 10
        status_2 = "✅ PASS" if canonical_pass else "❌ FAIL"
        click.echo(f"Canonical patterns found: {canonical_patterns} (≥10 required) {status_2}")
        
        # Criteria 3: Novel patterns discovered ≥ 5
        # Define canonical vs novel patterns based on infrastructure pattern taxonomy
        canonical_base_patterns = {
            'Replicate', 'Throttle', 'Scale', 'Restart', 'Timeout', 
            'CircuitBreaker', 'Retry', 'LoadBalance', 'Cache', 'Monitor'
        }
        
        found_patterns = set(results.get('operation_distribution', {}).keys())
        canonical_found = found_patterns.intersection(canonical_base_patterns)
        novel_found = found_patterns - canonical_base_patterns
        
        # Count novel patterns (beyond the 10 basic canonical ones)
        novel_patterns = len(novel_found)
        canonical_patterns_actual = len(canonical_found)
        
        novel_pass = novel_patterns >= 5
        status_3 = "✅ PASS" if novel_pass else "❌ FAIL"
        click.echo(f"Novel patterns discovered: {novel_patterns} (≥5 required) {status_3}")
        
        # Show pattern breakdown
        if novel_found:
            click.echo(f"   Novel patterns found: {', '.join(sorted(novel_found))}")
        if canonical_found:
            click.echo(f"   Canonical patterns: {', '.join(sorted(canonical_found))}")
        
        # Criteria 4: Runtime < 6 hours
        runtime_pass = runtime_hours < 6.0
        status_4 = "✅ PASS" if runtime_pass else "❌ FAIL"
        click.echo(f"Runtime: {runtime_hours:.2f}h (<6h required) {status_4}")
        
        # Overall Gate 3 status
        all_criteria_pass = accuracy_pass and canonical_pass and novel_pass and runtime_pass
        gate_status = "🎉 GATE 3 PASSED!" if all_criteria_pass else "🚫 GATE 3 FAILED"
        click.echo(f"\n{gate_status}")
        
        # Show detailed results
        click.echo(f"\n📊 Detailed Results:")
        click.echo(f"   Total scenarios: {results['total_scenarios']}")
        click.echo(f"   Lines matched: {results['matched_lines']}/{results['total_lines']}")
        click.echo(f"   Runtime: {runtime_hours:.2f} hours")
        
        # CRITICAL WARNING about circular validation
        if results['total_lines'] > 100:  # Synthetic data has many lines
            click.echo(f"\n⚠️  CIRCULAR VALIDATION WARNING:")
            click.echo(f"   Validating {results['total_lines']} synthetic config lines")
            click.echo(f"   against the same patterns used to generate them.")
            click.echo(f"   This artificially inflates accuracy to {accuracy:.1f}%.")
            click.echo(f"   For Gate 4: Need external validation dataset!")
        
        # Show top operations
        if results.get('operation_distribution'):
            click.echo(f"\n🔧 Top Operations Found:")
            sorted_ops = sorted(results['operation_distribution'].items(), key=lambda x: x[1], reverse=True)
            for op, count in sorted_ops[:10]:
                click.echo(f"   {op}: {count} occurrences")
        
        # Failure guidance
        if not all_criteria_pass:
            click.echo(f"\n💡 Troubleshooting:")
            if not accuracy_pass:
                click.echo(f"   • Low accuracy: Check {diagnostics_dir}/unmatched_lines.log")
                click.echo(f"   • Consider extending regex patterns or enabling GPT fallback")
            if not canonical_pass:
                click.echo(f"   • Few patterns: Check pattern extraction logic")
            if not novel_pass:
                click.echo(f"   • Novel patterns: Implement proper novelty detection")
            if not runtime_pass:
                click.echo(f"   • Runtime too long: Consider reducing count to 400")
                click.echo(f"   • Or optimize pattern matching performance")
            
            click.echo(f"\n🔄 Suggested retry:")
            click.echo(f"   poetry run python protean/cli.py batch-generate --count 400")
        
        # Show diagnostics location
        click.echo(f"\n📋 Full diagnostics available in: {diagnostics_dir}")
        click.echo(f"📝 Scenarios written to: {outdir}")
        
        return results
        
    except Exception as e:
        click.echo(f"❌ Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise click.ClickException(str(e))


@main.command()
def pipeline() -> None:
    """Run the complete pattern discovery pipeline."""
    click.echo("🚀 Starting complete Protean pipeline")
    click.echo("📊 Week 1 Mission: Pattern Genome Discovery")
    
    # TODO: Implement full pipeline orchestration
    click.echo("1. 🔧 Synthetic generation (2 CPU-hours max)")
    click.echo("2. 🔍 Pattern extraction (3 CPU-hours max)")
    click.echo("3. ✅ Validation (1 CPU-hour max)")
    click.echo("4. 📊 Visualization")
    
    click.echo("✅ Pipeline complete - Pattern genome ready!")


@main.command("validate-external")
@click.option("--external-config", "-e", type=click.Path(exists=True), 
              default="data/external/external_config_lines.txt",
              help="Path to external configuration lines file")
@click.option("--ground-truth", "-g", type=click.Path(exists=True),
              default="data/external/ground_truth_labels.json", 
              help="Path to ground truth labels file")
@click.option("--output-dir", "-o", type=click.Path(), default="data/diagnostics",
              help="Output directory for validation results")
def validate_external(external_config: str, ground_truth: str, output_dir: str) -> None:
    """
    Validate against external dataset (REAL ACCURACY TEST)
    
    This validates our patterns against independent, real-world infrastructure 
    configurations that were NOT generated by our own patterns. This gives us
    genuine accuracy metrics and solves the circular validation problem.
    """
    click.echo("🔬 External validation - Testing against REAL infrastructure configs")
    click.echo(f"📂 External config: {external_config}")
    click.echo(f"📋 Ground truth: {ground_truth}")
    click.echo(f"📁 Output dir: {output_dir}")
    
    # Import here to avoid circular imports
    from protean.core.validator import ScenarioValidator
    
    try:
        # Initialize validator
        validator = ScenarioValidator(
            data_dir="data/external",  # Won't be used for external validation
            output_dir=output_dir,
            max_scenarios=50  # Won't be used for external validation
        )
        
        # Run external validation
        results = validator.validate_external_dataset(external_config, ground_truth)
        
        if not results['success']:
            click.echo(f"❌ External validation failed: {results.get('error', 'Unknown error')}")
            return
        
        # Report results
        pattern_coverage = results['pattern_coverage_accuracy']
        ground_truth_accuracy = results['ground_truth_accuracy']
        matched_lines = results['matched_lines']
        total_lines = results['total_lines']
        correct_predictions = results['correct_predictions']
        ground_truth_total = results['ground_truth_total']
        
        click.echo(f"\n🎯 REAL ACCURACY RESULTS (External Validation):")
        click.echo(f"{'='*55}")
        
        # Pattern coverage (how many lines we can classify)
        coverage_status = "✅ GOOD" if pattern_coverage >= 70 else "⚠️  LOW"
        click.echo(f"Pattern Coverage: {pattern_coverage:.1f}% ({matched_lines}/{total_lines} lines) {coverage_status}")
        
        # Ground truth accuracy (how many we classify correctly)
        if ground_truth_total > 0:
            truth_status = "✅ GOOD" if ground_truth_accuracy >= 70 else "⚠️  NEEDS WORK"
            click.echo(f"Classification Accuracy: {ground_truth_accuracy:.1f}% ({correct_predictions}/{ground_truth_total} correct) {truth_status}")
        else:
            click.echo(f"Classification Accuracy: N/A (no ground truth labels)")
        
        # Overall assessment
        click.echo(f"\n📊 Assessment:")
        if pattern_coverage >= 70 and ground_truth_accuracy >= 70:
            click.echo(f"🎉 EXCELLENT: Ready for production embeddings!")
        elif pattern_coverage >= 50:
            click.echo(f"🔧 GOOD: Patterns work but need refinement")
        else:
            click.echo(f"⚠️  NEEDS IMPROVEMENT: Low pattern coverage")
        
        # Show top matched patterns
        if results.get('operation_distribution'):
            click.echo(f"\n🔧 External Pattern Distribution:")
            sorted_ops = sorted(results['operation_distribution'].items(), key=lambda x: x[1], reverse=True)
            for op, count in sorted_ops[:8]:
                click.echo(f"   {op}: {count} occurrences")
        
        # Show diagnostics location
        click.echo(f"\n📋 Detailed diagnostics: {output_dir}")
        click.echo(f"   External matched: {output_dir}/external_matched_lines.log")
        click.echo(f"   External unmatched: {output_dir}/external_unmatched_lines.log")
        
    except Exception as e:
        click.echo(f"❌ External validation error: {e}")
        raise click.ClickException(str(e))


@main.command("train-embedder")
@click.option("--output-dir", "-o", type=click.Path(), default="protean/models",
              help="Output directory for trained model")
@click.option("--embedding-dim", type=int, default=256,
              help="Embedding dimension")
@click.option("--hidden-dim", type=int, default=512,
              help="Hidden dimension for LSTM")
@click.option("--epochs", type=int, default=100,
              help="Number of training epochs")
@click.option("--batch-size", type=int, default=32,
              help="Training batch size")
def train_embedder(output_dir: str, embedding_dim: int, hidden_dim: int, epochs: int, batch_size: int) -> None:
    """
    Train pattern embedder model from validated pattern data.
    
    Creates models/pattern_embedder.pt for Week 1 deliverable.
    Uses both internal validation data and external real configs.
    """
    click.echo("🤖 Training Pattern Embedder Model")
    click.echo("📊 Week 1 Deliverable: models/pattern_embedder.pt")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"🎯 Model config: {embedding_dim}D embeddings, {hidden_dim}D hidden, {epochs} epochs")
    
    try:
        # Import here to avoid circular imports
        from protean.models.embedder.pattern_embedder import train_pattern_embedder
        
        # Start training
        model_path = train_pattern_embedder(
            output_dir=output_dir,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_epochs=epochs,
            batch_size=batch_size
        )
        
        click.echo(f"✅ Pattern embedder training complete!")
        click.echo(f"🎯 Model saved: {model_path}")
        click.echo(f"📦 Week 1 deliverable ready: models/pattern_embedder.pt")
        
        # Show training summary
        import pickle
        from pathlib import Path
        
        metadata_path = Path(output_dir) / "pattern_embedder_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            results = metadata['training_results']
            click.echo(f"\n📊 Training Results:")
            click.echo(f"   Final validation accuracy: {results['final_metrics']['accuracy']:.1%}")
            click.echo(f"   Final validation F1 score: {results['final_metrics']['f1']:.3f}")
            click.echo(f"   Training time: {results['training_time']:.1f}s")
            click.echo(f"   Pattern vocabulary: {len(metadata['vocab'])} tokens")
            click.echo(f"   Pattern categories: {metadata['model_config']['num_patterns']}")
            click.echo(f"   Training instances: {metadata['num_instances']}")
        
    except Exception as e:
        click.echo(f"❌ Embedding training failed: {e}")
        import traceback
        traceback.print_exc()
        raise click.ClickException(str(e))


@main.command("train-embeddings")
@click.option("--patterns", "-p", type=click.Path(exists=True), required=True,
              help="Path to pattern graphs pickle file")
@click.option("--epochs", type=int, default=80,
              help="Number of training epochs")
@click.option("--batch-size", type=int, default=64,
              help="Training batch size")
@click.option("--learning-rate", type=float, default=0.001,
              help="Learning rate")
@click.option("--embedding-dim", type=int, default=512,
              help="Embedding dimension")
@click.option("--hidden-dim", type=int, default=1024,
              help="Hidden dimension")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Output path for trained model")
@click.option("--gpu-hours-budget", type=float, default=10.0,
              help="GPU hours budget")
@click.option("--target-loss", type=float, default=0.40,
              help="Target triplet loss")
def train_embeddings(patterns: str, epochs: int, batch_size: int, learning_rate: float,
                    embedding_dim: int, hidden_dim: int, output: str,
                    gpu_hours_budget: float, target_loss: float) -> None:
    """
    GPU-optimized embedding training with triplet loss.
    
    Advanced training for Lambda GPU instances with pattern graphs.
    Target: Final triplet loss <0.40 in 8-10 GPU hours.
    """
    click.echo("🚀 GPU-Optimized Pattern Embedding Training")
    click.echo(f"📊 Loading pattern graphs from: {patterns}")
    click.echo(f"🎯 Target: Triplet loss <{target_loss} in {gpu_hours_budget}h")
    click.echo(f"🤖 Config: {embedding_dim}D embeddings, {epochs} epochs, batch {batch_size}")
    
    try:
        # Import GPU training modules
        from protean.models.embedder.gpu_trainer import AdvancedPatternEmbedder, TripletLossTrainer
        import pickle
        import time
        import torch
        
        start_time = time.time()
        
        # Load pattern graphs
        with open(patterns, 'rb') as f:
            pattern_graphs = pickle.load(f)
        
        click.echo(f"✅ Loaded {len(pattern_graphs)} pattern graphs")
        
        # Initialize GPU trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            click.echo("⚠️  No GPU detected! Training will be slow.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            click.echo(f"🚀 Using GPU: {gpu_name}")
        
        # Create advanced model with triplet loss
        trainer = TripletLossTrainer(
            pattern_graphs=pattern_graphs,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            device=device,
            gpu_hours_budget=gpu_hours_budget
        )
        
        # Train with triplet loss
        results = trainer.train_with_triplet_loss(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            target_loss=target_loss,
            save_path=output
        )
        
        # Report results
        training_time = (time.time() - start_time) / 3600  # hours
        final_loss = results['final_triplet_loss']
        
        click.echo(f"\n🎯 Training Results:")
        click.echo(f"   Final triplet loss: {final_loss:.4f} (target: <{target_loss})")
        click.echo(f"   Training time: {training_time:.2f}h (budget: {gpu_hours_budget}h)")
        click.echo(f"   Epochs completed: {results['epochs_completed']}")
        click.echo(f"   Model saved: {output}")
        
        # Check success criteria
        loss_success = final_loss < target_loss
        time_success = training_time <= gpu_hours_budget
        
        if loss_success and time_success:
            click.echo("🎉 SUCCESS: All targets achieved!")
        elif loss_success:
            click.echo("⚠️  Target loss achieved but exceeded time budget")
        elif time_success:
            click.echo("⚠️  Within time budget but target loss not achieved")
        else:
            click.echo("❌ Failed to meet targets")
        
        # Show embedding quality metrics
        if 'embedding_metrics' in results:
            metrics = results['embedding_metrics']
            click.echo(f"\n📊 Embedding Quality:")
            click.echo(f"   Canonical cluster coherence: {metrics['canonical_coherence']:.3f}")
            click.echo(f"   Novel pattern separation: {metrics['novel_separation']:.3f}")
            click.echo(f"   Cross-validation accuracy: {metrics['cv_accuracy']:.3f}")
        
    except Exception as e:
        click.echo(f"❌ GPU embedding training failed: {e}")
        import traceback
        traceback.print_exc()
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main() 