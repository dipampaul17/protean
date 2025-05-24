#!/bin/bash
# Quick start script for Lambda training

echo "🚀 Protean Lambda Training Setup"
echo "================================="

# Test SSH connection
echo "🔗 Testing SSH connection..."
if ssh -o ConnectTimeout=5 ubuntu@159.54.183.176 "echo 'SSH working!'"; then
    echo "✅ SSH connection successful!"
    
    # Copy training script
    echo "📤 Copying training script..."
    scp lambda_training.py ubuntu@159.54.183.176:~/
    
    # Start training
    echo "🎯 Starting training on Lambda..."
    ssh ubuntu@159.54.183.176 "python3 lambda_training.py"
    
else
    echo "❌ SSH connection failed!"
    echo "Please add the SSH key and restart the instance:"
    echo ""
    echo "🔑 Public Key:"
    cat ~/.ssh/lambda_protean.pub
    echo ""
    echo "📍 Add it at: https://cloud.lambda.ai/ssh-keys"
    echo "🔄 Then restart instance: 159.54.183.176"
fi 