#ifndef COMMON_CUH
#define COMMON_CUH

#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


// Error checking macro: wraps any CUDA call and prints file, line, error code,
// and human-readable reason before aborting if the call fails.
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    const cudaError_t _error = (call);                                         \
    if (_error != cudaSuccess) {                                               \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code:%d, reason: %s\n", _error,                         \
              cudaGetErrorString(_error));                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Number of threads per block used throughout all kernels
#define BLOCK_SIZE 256

// --- DATA STRUCTURES ---

// Simple 3D point structure
struct Point3D {
  float x, y, z;
};

// --- COMMON FUNCTIONS ---

// Device/Host function to compute Euclidean distance between two 3D points
__device__ __host__ inline float euclidean_distance(const Point3D &a,
                                                    const Point3D &b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  float dz = a.z - b.z;
  return sqrtf(dx * dx + dy * dy + dz * dz);
}

// Function to load the CSV dataset
// Reads points from a CSV file (skips first row, first column)
inline int load_dataset(const std::string &filename,
                        std::vector<Point3D> &points) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Impossible to open file " << filename << std::endl;
    return 0;
  }

  std::string line;
  // Skip the header
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    Point3D p = {0.0f, 0.0f, 0.0f};

    // CSV format: source_id, x, y, z, ...
    // Skip source_id
    std::getline(ss, cell, ',');

    // Read X
    std::getline(ss, cell, ',');
    try {
      p.x = std::stof(cell);
    } catch (...) {
      p.x = 0.0f;
    }

    // Read Y
    std::getline(ss, cell, ',');
    try {
      p.y = std::stof(cell);
    } catch (...) {
      p.y = 0.0f;
    }

    // Read Z
    std::getline(ss, cell, ',');
    try {
      p.z = std::stof(cell);
    } catch (...) {
      p.z = 0.0f;
    }

    points.push_back(p);
  }

  int num_points = points.size();
  printf("Read %d points from CSV.\n", num_points);
  return num_points;
}

#endif // COMMON_CUH
