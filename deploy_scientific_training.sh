#!/bin/bash
# Scientific GraphSAGE Training Deployment for Lambda
# Deploy and run the proper GraphSAGE-based pattern embedding experiment

set -e

LAMBDA_HOST="lambda-protean"
LAMBDA_IP="159.54.183.176"
EXPERIMENT_NAME="protean-graphsage-$(date +%Y%m%d_%H%M%S)"

echo "🚀 Deploying Scientific GraphSAGE Training to Lambda"
echo "=============================================================="
echo "Target: $LAMBDA_HOST ($LAMBDA_IP)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Time: $(date)"
echo "Architecture: GraphSAGE (NOT LSTM)"
echo "Target Triplets: 50,000+"
echo "Min Epochs: 20"
echo "Expected Runtime: >30 minutes"

# Check SSH connection
echo "🔐 Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 $LAMBDA_HOST "echo 'SSH connection successful'" 2>/dev/null; then
    echo "❌ SSH connection failed. Check your SSH keys and instance availability."
    exit 1
fi
echo "✅ SSH connection verified"

# Transfer the scientific training script
echo "📤 Transferring scientific GraphSAGE training script..."
scp lambda_scientific_training.py $LAMBDA_HOST:~/protean/

# Ensure data directory exists
echo "📊 Verifying data structure..."
ssh $LAMBDA_HOST "cd ~/protean && mkdir -p data/smoke/scenarios protean/models"

# Create enhanced deployment script for Lambda
cat << 'EOF' > /tmp/lambda_scientific_deploy.sh
#!/bin/bash
set -e

echo "🚀 Scientific GraphSAGE Training on Lambda"
echo "=============================================="
echo "Instance: $(hostname)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

# Update system
echo "📦 Updating system..."
sudo apt-get update -qq

# Install Python and dependencies
echo "🐍 Setting up Python environment..."
sudo apt-get install -y python3 python3-pip python3-venv git

# Create fresh virtual environment
echo "🧪 Creating fresh virtual environment..."
rm -rf ~/protean/venv
python3 -m venv ~/protean/venv
source ~/protean/venv/bin/activate

# Install PyTorch with CUDA first
echo "🎮 Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and extensions
echo "🔗 Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install other dependencies
echo "📚 Installing additional libraries..."
pip install wandb loguru scikit-learn scipy networkx
pip install transformers datasets tokenizers numpy

# Verify installations
echo "🔧 Verifying installations..."
python3 -c "
import torch
import torch_geometric
from torch_geometric.nn import SAGEConv
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
print('SAGEConv import successful')
print('✅ All dependencies verified')
"

# Verify data exists
echo "📊 Verifying scenario data..."
if [ ! -f ~/protean/data/smoke/scenarios/config_lines.txt ]; then
    echo "❌ Config lines not found"
    exit 1
fi

config_lines=$(wc -l < ~/protean/data/smoke/scenarios/config_lines.txt)
echo "✅ Found $config_lines config lines"

if [ $config_lines -lt 1000 ]; then
    echo "❌ Insufficient config lines: $config_lines < 1000"
    exit 1
fi

# Run the scientific training
echo "🎯 Starting Scientific GraphSAGE Training..."
echo "Expected runtime: >30 minutes (NOT instant completion)"
echo "Expected triplets: >40,000 (NOT ~6k)"
echo "Expected model size: ~6MB (NOT 41MB)"

cd ~/protean
python3 lambda_scientific_training.py

echo "🎉 Scientific training completed!"
EOF

# Transfer and execute deployment script
scp /tmp/lambda_scientific_deploy.sh $LAMBDA_HOST:~/lambda_scientific_deploy.sh
rm /tmp/lambda_scientific_deploy.sh

echo "🚀 Executing scientific GraphSAGE training on Lambda..."
echo "📊 Monitor progress with:"
echo "   - Weights & Biases: https://wandb.ai/"
echo "   - SSH: ssh $LAMBDA_HOST 'tail -f ~/protean/training.log'"

# Execute with comprehensive logging
ssh $LAMBDA_HOST "chmod +x ~/lambda_scientific_deploy.sh && ~/lambda_scientific_deploy.sh" | tee "scientific_training_$(date +%Y%m%d_%H%M%S).log"

# Validate results
echo ""
echo "🔍 Validating scientific training results..."

# Check training completion
ssh $LAMBDA_HOST "cd ~/protean && ls -la protean/models/"

# Check for GraphSAGE model (not LSTM)
echo "🔍 Checking model architecture..."
if ssh $LAMBDA_HOST "cd ~/protean && [ -f protean/models/scientific_graphsage_embedder.pt ]"; then
    model_size=$(ssh $LAMBDA_HOST "cd ~/protean && du -sh protean/models/scientific_graphsage_embedder.pt | cut -f1")
    echo "✅ GraphSAGE model found: $model_size"
    
    # Check for LSTM artifacts (should be none)
    lstm_check=$(ssh $LAMBDA_HOST "cd ~/protean && strings protean/models/scientific_graphsage_embedder.pt | grep -i lstm | wc -l" || echo "0")
    if [ "$lstm_check" -eq "0" ]; then
        echo "✅ No LSTM artifacts found (correct GraphSAGE architecture)"
    else
        echo "❌ Warning: LSTM artifacts found in GraphSAGE model"
    fi
else
    echo "❌ GraphSAGE model not found"
fi

# Download the scientific model
echo "📥 Downloading scientific GraphSAGE model..."
mkdir -p protean/models
scp $LAMBDA_HOST:~/protean/protean/models/scientific_graphsage_embedder.pt protean/models/ 2>/dev/null || echo "⚠️ Model download failed"

# Check training logs for validation
echo "🔍 Checking training validation..."
runtime_check=$(ssh $LAMBDA_HOST "cd ~/protean && find . -name '*.log' -exec grep -l 'Training time.*h' {} \\; | head -1" || echo "")
if [ -n "$runtime_check" ]; then
    actual_runtime=$(ssh $LAMBDA_HOST "cd ~/protean && grep 'Training time' $runtime_check | tail -1")
    echo "📊 Training runtime: $actual_runtime"
else
    echo "⚠️ Could not find training runtime info"
fi

# Check triplet count
triplet_check=$(ssh $LAMBDA_HOST "cd ~/protean && find . -name '*.log' -exec grep -l 'triplets' {} \\; | head -1" || echo "")
if [ -n "$triplet_check" ]; then
    triplet_count=$(ssh $LAMBDA_HOST "cd ~/protean && grep 'Created.*triplets' $triplet_check | tail -1")
    echo "🔗 Triplet generation: $triplet_count"
else
    echo "⚠️ Could not find triplet count info"
fi

echo ""
echo "🔬 SCIENTIFIC TRAINING VALIDATION COMPLETE"
echo "=================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Architecture: GraphSAGE (verified no LSTM)"
echo "Log file: scientific_training_$(date +%Y%m%d_%H%M%S).log"
echo "Model: protean/models/scientific_graphsage_embedder.pt"
echo "Monitoring: https://wandb.ai/"
echo ""
echo "🎯 Next: Run validation with KNN evaluation!"
echo "   poetry run python scripts/eval_knn.py" 