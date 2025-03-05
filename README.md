# Origami-Nets: Neural Networks Inspired by Geometric Folding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Origami-Nets introduces a novel neural network architecture inspired by the principles of geometric folding and the Fold and Cut Theorem. Rather than relying solely on increasing model size to improve performance, we introduce the **Fold Layer** - a parameter-efficient alternative to traditional neural network layers that achieves comparable or better performance with significantly fewer parameters.

The core innovation is a learnable geometric transformation that "folds" data across hyperplanes, creating more efficiently separable representations for downstream classification or regression tasks.

## Key Features

- **Parameter Efficiency**: Fold layers scale as O(n) in parameters compared to O(n×m) for traditional linear layers
- **Reduced Inference Time**: Fewer parameters means faster forward pass computation
- **Improved Interpretability**: Fold operations provide a geometric interpretation of network transformations
- **Flexible Integration**: Fold layers can be incorporated into existing neural network architectures as drop-in replacements for linear layers
- **Better Scaling Properties**: Maintain performance with smaller models, ideal for resource-constrained environments

## How It Works

The Fold Layer operates by defining a hyperplane in an n-dimensional space and reflecting points across this hyperplane. Mathematically, for an input vector x and a learnable normal vector n defining the hyperplane:

1. The fold layer identifies points on one side of the hyperplane
2. It applies a reflection transformation to those points
3. This operation effectively "folds" the data space, making complex decision boundaries easier to learn

We provide several variations:
- **Hard Fold**: Uses a sharp boundary for folding
- **Leaky Fold**: Allows gradients to propagate through non-folded regions
- **Soft Fold**: Implements a smooth transition across the fold using a sigmoid function

## Results

Our experiments across various datasets demonstrate that Origami-Nets can:

- Match or exceed the performance of traditional MLPs with fewer parameters
- Reduce model size by up to 90% without sacrificing accuracy
- Accelerate inference time for real-time applications
- Provide more parameter-efficient alternatives for reinforcement learning tasks

## Applications

Origami-Nets are particularly suitable for:

- Resource-constrained environments (mobile/edge devices)
- Real-time inference applications
- Large-scale models where parameter efficiency is crucial
- Applications where model interpretability is important

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/Origami-Nets.git
cd Origami-Nets
pip install -r requirements.txt
```

### Basic Usage

```python
from models.model_bank import OrigamiFold4
from models.training import *

# Load your data
# x_train, y_train, x_test, y_test = ...

# Create data loaders
train_loader = load_data(x_train, y_train)
val_loader = load_data(x_test, y_test)

# Initialize model
model = OrigamiFold4(x_train.shape[1])

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model, optimizer, train_loader, val_loader, epochs=200, verbose=1)
```

## Research Background

This work addresses the challenge of improving neural network efficiency without sacrificing performance. Traditional approaches focus on increasing model size to leverage neural scaling laws, but this comes with high computational costs and inefficiency.

Inspired by origami and the Fold and Cut Theorem, our approach demonstrates that geometrically motivated operations can achieve complex decision boundaries with far fewer parameters than conventional methods.

The theoretical foundation combines concepts from computational geometry, manifold learning, and the study of decision boundaries in high-dimensional spaces.

## Contributors

- Sam Layton
- Dallin Stewart - dallinpstewart@gmail.com
- Jeddy Bennett - jeddybennett01@gmail.com

## Citation

If you use Origami-Nets in your research, please cite our work:

```
@article{origami_nets2023,
  title={Origami Inspired Neural Networks},
  author={Layton, Sam and Stewart, Dallin and Bennett, Jeddy},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
