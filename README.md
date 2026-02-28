# Parallel Clustering and Minimum Spanning Tree Algorithms

**Authors**: Bonsignore Christian, Ferrara Ludovico, Toma Elia

## Overview
This project implements high-performance parallel algorithms for astronomical data analysis, specifically focusing on density-based clustering and hierarchical single-linkage clustering. The primary objective is to evaluate the performance and accuracy of GPU-accelerated implementations against their sequential counterparts.

## Project Structure
- `PDPProject.ipynb`: Main experimentation and benchmarking notebook.
- `optimized_alg/`: Contains the optimized CUDA implementations (`gdbscan.cu`, `boruvka.cu`) and shared header extracted from the original implementation (`common.cuh`).
- `original_alg/`: Reference implementations (`gdbscan.cu`, `parallel_slink.cu`).
- `dataset`: Dataset storage and management (`data/PDPProject/DATASET`).
- `utility`: Utility functions used to produce the output files (`data/PDPProject/UTILS`).
- `slink`: Slink implementation taken from the original repository (`data/PDPProject/SLINK`).
- `output`: Output .csv files (`data/PDPProject/OUTPUT`).

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