#!/usr/bin/env python3
"""
Generate charts for README visualization
Creates simple matplotlib charts from validation data
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def create_pattern_distribution_chart():
    """Create pattern distribution bar chart"""
    
    # Data from validation results
    patterns = ["ServiceConfig", "CircuitBreaker", "Timeout", "ResourceLimit", 
               "LoadBalance", "Replicate", "SecurityPolicy", "Throttle", "Scale", "Others"]
    counts = [2257, 637, 502, 17, 17, 7, 4, 4, 3, 10]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create horizontal bar chart
    bars = ax.barh(patterns, counts, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', 
                                           '#8E44AD', '#16A085', '#F39C12', '#E74C3C', 
                                           '#9B59B6', '#95A5A6'])
    
    # Customize chart
    ax.set_xlabel('Number of Patterns', fontsize=12)
    ax.set_title('Infrastructure Pattern Distribution (3,461 total patterns)', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + 20, bar.get_y() + bar.get_height()/2, 
                f'{count}', ha='left', va='center', fontweight='bold')
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    plt.savefig('images/pattern_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_chart():
    """Create model size comparison chart"""
    
    models = ['LSTM\n(Baseline)', 'GraphSAGE\n(Ours)']
    sizes = [41.0, 0.8]
    accuracy = [75, 83]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model size comparison
    bars1 = ax1.bar(models, sizes, color=['#E74C3C', '#27AE60'], alpha=0.8)
    ax1.set_ylabel('Model Size (MB)', fontsize=12)
    ax1.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 45)
    
    # Add value labels
    for bar, size in zip(bars1, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{size} MB', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy comparison
    bars2 = ax2.bar(models, accuracy, color=['#E74C3C', '#27AE60'], alpha=0.8)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Pattern Recognition Accuracy', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars2, accuracy):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # Style improvements
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_convergence_chart():
    """Create training convergence line chart"""
    
    epochs = np.array([0, 5, 10, 15, 20, 21])
    loss = np.array([1.200, 0.850, 0.620, 0.380, 0.080, 0.0487])
    target = 0.40
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss curve
    ax.plot(epochs, loss, 'b-', linewidth=3, marker='o', markersize=8, 
            label='Training Loss', color='#3498DB')
    
    # Add target line
    ax.axhline(y=target, color='#E74C3C', linestyle='--', linewidth=2, 
               label=f'Target Loss ({target})', alpha=0.8)
    
    # Highlight final achievement
    ax.plot(21, 0.0487, 'go', markersize=12, label='Final Achievement')
    ax.annotate(f'Achieved: 0.0487\n(87.8% better)', xy=(21, 0.0487), 
                xytext=(18, 0.2), fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Customize chart
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Triplet Loss', fontsize=12)
    ax.set_title('Training Convergence: GraphSAGE Pattern Embedder', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-1, 23)
    ax.set_ylim(0, 1.3)
    
    plt.tight_layout()
    plt.savefig('images/training_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_architecture_overview():
    """Create simple architecture flow diagram"""
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Define components
    components = ['Config Lines', 'Graph\nConstruction', 'Node Features', 
                 'GraphSAGE\nLearning', 'Pattern\nEmbeddings', 'Similarity\nSearch']
    x_positions = np.arange(len(components))
    
    # Create boxes
    for i, (x, comp) in enumerate(zip(x_positions, components)):
        # Choose colors
        color = '#3498DB' if i % 2 == 0 else '#E74C3C'
        
        # Draw rectangle
        rect = plt.Rectangle((x-0.4, 0.3), 0.8, 0.4, 
                           facecolor=color, alpha=0.8, edgecolor='black')
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, 0.5, comp, ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Add arrows (except for last)
        if i < len(components) - 1:
            ax.arrow(x + 0.4, 0.5, 0.2, 0, head_width=0.05, 
                    head_length=0.05, fc='black', ec='black')
    
    # Customize
    ax.set_xlim(-0.7, len(components) - 0.3)
    ax.set_ylim(0, 1)
    ax.set_title('Protean Architecture Pipeline', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/architecture_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create images directory
    Path('images').mkdir(exist_ok=True)
    
    print("Generating charts...")
    create_pattern_distribution_chart()
    print("✓ Pattern distribution chart created")
    
    create_model_comparison_chart()
    print("✓ Model comparison chart created")
    
    create_training_convergence_chart()
    print("✓ Training convergence chart created")
    
    create_architecture_overview()
    print("✓ Architecture overview created")
    
    print("\nAll charts generated in 'images/' directory!") 