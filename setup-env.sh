#!/bin/bash
set -e  # stop on first error

ENV_NAME="John_RL"
PYTHON_VERSION=3.11

echo "====================================="
echo "🚀 Creating Conda RL environment"
echo "====================================="
# create env
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# activate env (important: must use source)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME



echo "====================================="
echo "📦 Upgrading pip tools"
echo "====================================="
python -m pip install --upgrade pip




echo "🚀 Creating RL environment setup..."
# upgrade pip tools
pip install --upgrade pip setuptools==81.0.0 wheel==0.46.3

echo "🎮 Installing Gymnasium + MuJoCo..."
pip install "gymnasium[mujoco]==1.2.3"

echo "🧠 Installing Stable-Baselines3..."
pip install "stable-baselines3[extra]==2.8.0"

echo "📊 Installing utilities..."
pip install imageio[ffmpeg] notebook



echo "====================================="
echo "✅ Environment ready!"
echo "👉 To activate later:"
echo "   conda activate $ENV_NAME"
echo "====================================="

