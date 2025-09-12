#!/bin/bash
# Fix NumPy/SciPy/TensorFlow compatibility issues

echo "🔧 Fixing NumPy/SciPy/TensorFlow Compatibility Issues"
echo "=" * 60

# Check current versions
echo "Current package versions:"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: Not available"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')" 2>/dev/null || echo "SciPy: Not available"
python3 -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')" 2>/dev/null || echo "TensorFlow: Not available"

echo ""
echo "🔄 Installing compatible versions..."

# Remove problematic packages
pip3 uninstall -y numpy scipy tensorflow tensorflow-cpu

# Install compatible versions in correct order
echo "📦 Installing NumPy 1.21.6..."
pip3 install numpy==1.21.6

echo "📦 Installing SciPy 1.7.3..."
pip3 install scipy==1.7.3

echo "📦 Installing TensorFlow 2.10.1..."
pip3 install tensorflow==2.10.1

echo "📦 Installing other dependencies..."
pip3 install gym==0.26.2
pip3 install matplotlib

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Testing imports..."
python3 -c "
import numpy as np
import scipy
import tensorflow as tf
print(f'✓ NumPy {np.__version__}')
print(f'✓ SciPy {scipy.__version__}')
print(f'✓ TensorFlow {tf.__version__}')
print('✅ All imports successful!')
"

echo ""
echo "🚀 Ready to run training!"
echo "Try: python3 main.py --mode train --no-robot --episodes 10"
