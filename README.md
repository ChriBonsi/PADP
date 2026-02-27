# High-Performance Parallel Clustering and Minimum Spanning Tree Algorithms

**Authors**: Bonsignore Christian, Ferrara Ludovico, Toma Elia

## Overview
This project implements high-performance parallel algorithms for astronomical data analysis, specifically focusing on density-based clustering and hierarchical single-linkage clustering. The primary objective is to evaluate the performance and accuracy of GPU-accelerated implementations against their sequential counterparts.

## Key Algorithms

### G-DBSCAN (GPU Density-Based Spatial Clustering of Applications with Noise)
An optimized implementation of the DBSCAN algorithm leveraging NVIDIA CUDA. It features:
- **CSR Adjacency Representation**: Efficient storage of neighborhood relationships.
- **Shared Memory Tiling**: Cooperative loading of point tiles into shared memory to maximize bandwidth and reduce global memory access latency.
- **BFS Frontier Propagation**: Parallel cluster expansion using a double-buffered (ping-pong) BFS approach.

### Borůvka-Elias EMST (Euclidean Minimum Spanning Tree)
A parallel implementation for computing the EMST, which serves as the foundation for Single-Linkage Hierarchical Clustering. Key features include:
- **3D Cellular Grid Acceleration**: Employs a uniform grid to localize point neighbor searches, significantly pruning the search space.
- **Parallel Borůvka Iterations**: Concurrent component merging using atomic operations and pointer jumping for path compression.
- **Thrust Integration**: Utilizes the Thrust library for high-level memory management (RAII) and fundamental primitives likes scans and sorts.

## Project Structure
- `final_alg/`: Contains the optimized CUDA implementations (`dbscan2.cu`, `boruvka.cu`) and shared headers (`common.cuh`).
- `original_alg/`: Reference and legacy implementations (`gdbscan.cu`, `cuslink.cu`).
- `data/`: Dataset storage and management (`PDPProject/DATASET`).
- `PDPProject.ipynb`: Main experimentation and benchmarking notebook.

## Setup and Usage

### Data Preparation
1. Clone the repository:
   ```bash
   git clone https://github.com/ChriBonsi/PADP.git
   ```
2. Unzip the archives located in `data/PDPProject/DATASET/` into the same directory.
3. Ensure the result is a flat folder structure in `data/PDPProject/DATASET/` containing the `.csv` files.

### Execution on Google Colab
1. Upload the `PDPProject.ipynb` file to Google Colab.
2. Upload the entire `data/PDPProject` folder to your Google Drive to ensure persistent access to the datasets.
3. Follow the notebook cells to compile and run the algorithms in a GPU-enabled environment.