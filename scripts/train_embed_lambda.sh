#!/bin/bash
# Lambda GPU Training Script for Pattern Embeddings
# Target: 8-10 GPU hours, Final triplet loss <0.40

set -e

echo "🚀 Protean Pattern Embedder - Lambda GPU Training"
echo "🎯 Target: Final triplet loss <0.40"
echo "⏱️  Budget: 8-10 GPU hours"

# Submit Lambda job
lambdaprompt run \
  --instance-type a10g.small \
  --disk 20 \
  --timeout 7200 \
  --command "
set -e
echo '📦 Installing dependencies...'
poetry install --no-interaction

echo '🔧 Building pattern graphs...'
poetry run python scripts/build_pattern_graphs.py

echo '🤖 Starting GPU embedding training...'
poetry run python protean/cli.py train-embeddings \
  --patterns data/synthetic/pattern_graphs.pkl \
  --epochs 80 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --embedding-dim 512 \
  --hidden-dim 1024 \
  --output protean/models/pattern_embedder.pt \
  --gpu-hours-budget 10 \
  --target-loss 0.40

echo '✅ Training complete! Uploading results...'
ls -la protean/models/
"

echo "🎉 Lambda job submitted!"
echo "📊 Monitor logs for:"
echo "   ✅ Final triplet loss <0.40" 
echo "   ⏱️  Training time <10 GPU hours"
echo "   📦 Model saved to protean/models/pattern_embedder.pt" 