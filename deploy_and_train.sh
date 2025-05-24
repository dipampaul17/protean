#!/bin/bash
# Scientific Enhanced GPU Training Deployment for Lambda
# Deploy and run the enhanced pattern embedding experiment

set -e

LAMBDA_HOST="lambda-protean"
LAMBDA_IP="159.54.183.176"
EXPERIMENT_NAME="protean-enhanced-$(date +%Y%m%d_%H%M%S)"

echo "🚀 Deploying Enhanced Pattern Embedding Experiment to Lambda"
echo "=============================================================="
echo "Target: $LAMBDA_HOST ($LAMBDA_IP)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Time: $(date)"

# Check SSH connection
echo "🔐 Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 $LAMBDA_HOST "echo 'SSH connection successful'" 2>/dev/null; then
    echo "❌ SSH connection failed. Check your SSH keys and instance availability."
    exit 1
fi
echo "✅ SSH connection verified"

# Create project directory on Lambda
echo "📁 Setting up project directory..."
ssh $LAMBDA_HOST "mkdir -p ~/protean && cd ~/protean"

# Transfer essential files
echo "📤 Transferring enhanced training files..."

# Transfer the enhanced training script
scp lambda_enhanced_training.py $LAMBDA_HOST:~/protean/

# Transfer scenario data
echo "📊 Transferring scenario data..."
scp -r data/smoke/scenarios/config_lines.txt $LAMBDA_HOST:~/protean/config_lines.txt

# Transfer project metadata
scp pyproject.toml $LAMBDA_HOST:~/protean/

# Create deployment script for Lambda
cat << 'EOF' > /tmp/lambda_deploy.sh
#!/bin/bash
set -e

echo "🚀 Enhanced Scientific GPU Training on Lambda"
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

# Create virtual environment
python3 -m venv ~/protean/venv
source ~/protean/venv/bin/activate

# Install PyTorch with CUDA
echo "🎮 Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "📚 Installing ML libraries..."
pip install wandb loguru scikit-learn scipy networkx pathlib2
pip install transformers datasets tokenizers numpy

# Verify GPU setup
echo "🔧 Verifying GPU setup..."
python3 -c "
import sys
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('❌ No GPU detected!')
    sys.exit(1)
"

# Create data directory structure
echo "🏗️ Setting up data structure..."
mkdir -p ~/protean/data/smoke/scenarios
mkdir -p ~/protean/protean/models

# Move config file to proper location
mv ~/protean/config_lines.txt ~/protean/data/smoke/scenarios/

# Run the enhanced training
echo "🎯 Starting Enhanced Scientific Training..."
cd ~/protean
python3 lambda_enhanced_training.py

echo "🎉 Enhanced training completed!"
EOF

# Transfer and execute deployment script
scp /tmp/lambda_deploy.sh $LAMBDA_HOST:~/lambda_deploy.sh
rm /tmp/lambda_deploy.sh

echo "🚀 Executing enhanced training on Lambda..."
echo "📊 Monitor progress with Weights & Biases: https://wandb.ai/"

# Execute the training with output streaming
ssh $LAMBDA_HOST "chmod +x ~/lambda_deploy.sh && ~/lambda_deploy.sh" | tee "lambda_training_$(date +%Y%m%d_%H%M%S).log"

# Check results
echo ""
echo "🔍 Checking training results..."
ssh $LAMBDA_HOST "cd ~/protean && ls -la protean/models/ && echo '--- Training Log Tail ---' && tail -20 lambda_enhanced_training.log 2>/dev/null || echo 'No log file found'"

# Download trained model
echo "📥 Downloading trained model..."
mkdir -p protean/models
scp $LAMBDA_HOST:~/protean/protean/models/enhanced_pattern_embedder.pt protean/models/ 2>/dev/null || echo "⚠️ Model file not found (training may have failed)"

echo ""
echo "🔬 SCIENTIFIC EXPERIMENT COMPLETED"
echo "=================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Log file: lambda_training_$(date +%Y%m%d_%H%M%S).log"
echo "Model location: protean/models/enhanced_pattern_embedder.pt"
echo "Monitoring: https://wandb.ai/"
echo ""
echo "🎯 Check the log file and Weights & Biases dashboard for detailed results!" 