#!/bin/bash
# Lambda GPU Training Script with Weights & Biases Monitoring
# Run this on your Lambda GPU instance: ubuntu@159.54.183.176

set -e

echo "🚀 Protean GPU Training with Weights & Biases"
echo "==============================================="
echo "GPU Instance: $(hostname)"
echo "Date: $(date)"

# Set up environment
export WANDB_API_KEY="5379ff74a6e1ccca55c7c830d999e174ff971cc3"
export CUDA_VISIBLE_DEVICES=0

# Check GPU
echo "🎮 Checking GPU..."
nvidia-smi

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv

# Clone repo if needed (or update)
if [ ! -d "protean" ]; then
    echo "📥 Cloning Protean repository..."
    git clone https://github.com/yourusername/protean.git
    cd protean
else
    echo "📥 Updating Protean repository..."
    cd protean
    git pull
fi

# Set up Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install wandb loguru scikit-learn scipy networkx click pathlib
pip install transformers datasets tokenizers

# Login to wandb
echo "🎯 Setting up Weights & Biases..."
wandb login $WANDB_API_KEY

# Verify GPU setup
echo "🔧 Verifying GPU setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('❌ No GPU detected!')
    exit(1)
"

# Build pattern graphs
echo "🔧 Building pattern graphs..."
python3 scripts/build_pattern_graphs.py

# Start GPU training with wandb
echo "🚀 Starting GPU training with Weights & Biases monitoring..."
python3 -c "
import sys
sys.path.append('.')
import pickle
import torch
from protean.protean.models.embedder.gpu_trainer_wandb import WandbTripletLossTrainer

print('🔧 Loading pattern graphs...')
with open('data/synthetic/pattern_graphs.pkl', 'rb') as f:
    pattern_graphs = pickle.load(f)

print(f'✅ Loaded {len(pattern_graphs)} pattern graphs')

# Initialize GPU trainer with wandb
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'🎮 Using device: {device}')

trainer = WandbTripletLossTrainer(
    pattern_graphs=pattern_graphs,
    embedding_dim=512,
    hidden_dim=1024,
    device=device,
    gpu_hours_budget=10.0,
    wandb_project='protean-embeddings'
)

print('🎯 Starting training with Weights & Biases monitoring...')
results = trainer.train_with_triplet_loss_wandb(
    epochs=80,
    batch_size=64,
    learning_rate=0.001,
    target_loss=0.40,
    save_path='protean/models/pattern_embedder.pt',
    wandb_api_key='$WANDB_API_KEY'
)

print('✅ Training completed!')
print(f'Final triplet loss: {results[\"final_triplet_loss\"]:.4f}')
print(f'Target achieved: {results[\"target_achieved\"]}')
print(f'Training time: {results[\"training_time_hours\"]:.2f}h')
print(f'Epochs completed: {results[\"epochs_completed\"]}')

# Check if target achieved
if results['target_achieved']:
    print('🎉 SUCCESS: Target loss <0.40 achieved!')
else:
    print('⚠️ Target loss not achieved, but training completed')
"

echo "🎉 Training script completed!"
echo "📊 Check Weights & Biases dashboard for real-time metrics"
echo "🔗 https://wandb.ai/your-username/protean-embeddings" 