# JAX JIT and torch.compile Analysis

## Overview
This project analyses performance characteristics of JAX JIT compilation and PyTorch 2.0 `torch.compile`.

Experiments include:
- JAX compilation overhead
- Shape specialization behaviour
- Operator fusion performance
- PyTorch compilation backends
- Debugging compilation failures
- Graph capture analysis

## Requirements

Python 3.10+

Libraries:

pip install jax jaxlib torch matplotlib numpy



CUDA compatible PyTorch

## Files

1. jax_jit_analysis.ipynb  
   Experiments for JAX JIT compilation.

2. torch_compile_analysis.ipynb  
   Experiments for PyTorch torch.compile.

3. report.pdf  
   Final analysis report with explanations and plots.

## Running the notebooks

Open Jupyter or Google Colab.

Run in order:

1. jax_jit_analysis.ipynb  
2. torch_compile_analysis.ipynb

All plots and results will be generated automatically.

## Author
- Xavier S Adettu 
- COMPSCI 790
