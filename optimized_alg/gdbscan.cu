// dbscan2.cu — G-DBSCAN (improved version)
// Combines the modularity of dbscan.cu with the efficiency of dbscan1.cu.
// Key improvements:
//   - Thrust throughout (RAII, no manual cudaFree)
//   - Minimal D->H transfer for total_edges (only 2 integers)
//   - Graph struct for modularity
//   - CHECK_CUDA on all critical points
//   - Correct frontier reset (pre-swap) with clear of output buffer only
//   - Explicit synchronization with error checking

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#include "/content/drive/MyDrive/PDPProject/UTILS/common.cuh"

// ---------------------------------------------------------------------------
// DATA STRUCTURES
// ---------------------------------------------------------------------------

// Graph in CSR format: all vectors are device_vectors (automatic RAII)
struct Graph {
  thrust::device_vector<int> row_ptr; // row pointers: row_ptr[i]..row_ptr[i+1]
                                      // spans neighbors of point i
  thrust::device_vector<int>
      col_indices; // column indices: neighbor point indices
};

// ---------------------------------------------------------------------------
// KERNELS
// ---------------------------------------------------------------------------

// Counts the neighbors of each point (degrees), used to build row_ptr.
// Uses shared memory tiling: each block loads a tile of points into shared
// memory so that all threads in the block can reuse those reads together.
__global__ __launch_bounds__(BLOCK_SIZE) void count_neighbors_kernel(
    const Point3D *__restrict__ points, int *__restrict__ degrees,
    int num_points, float eps) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ Point3D
      sh_points[BLOCK_SIZE]; // shared memory tile for current block

  Point3D my_point;
  if (idx < num_points)
    my_point = points[idx];

  int count = 0;
  int num_tiles = (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Iterate over all tiles of points to count neighbors within eps
  for (int t = 0; t < num_tiles; ++t) {
    // Cooperatively load the current tile into shared memory
    int tile_idx = t * BLOCK_SIZE + threadIdx.x;
    if (tile_idx < num_points)
      sh_points[threadIdx.x] = points[tile_idx];
    __syncthreads(); // ensure tile is fully loaded before any thread reads it

    if (idx < num_points) {
      int num_in_tile = min(BLOCK_SIZE, num_points - t * BLOCK_SIZE);
      for (int j = 0; j < num_in_tile; ++j) {
        if (euclidean_distance(my_point, sh_points[j]) <= eps)
          count++;
      }
    }
    __syncthreads(); // ensure all threads are done before next tile is loaded
  }

  if (idx < num_points)
    degrees[idx] = count;
}

// KERNEL 1: Builds the adjacency list (col_indices) in CSR format.
// Each thread writes the neighbor indices of its assigned point starting at
// row_ptr[idx], which was pre-computed by prefix sum over degrees.
__global__ __launch_bounds__(BLOCK_SIZE) void construct_adjacency_list(
    const Point3D *__restrict__ points, const int *__restrict__ row_ptr,
    int *__restrict__ col_indices, int num_points, float eps) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ Point3D
      sh_points[BLOCK_SIZE]; // shared memory tile for current block

  Point3D my_point;
  int start_offset = 0, count = 0;

  if (idx < num_points) {
    my_point = points[idx];
    start_offset = row_ptr[idx]; // starting write position in col_indices
  }

  int num_tiles = (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Iterate over all tiles; write neighbor indices into col_indices
  for (int t = 0; t < num_tiles; ++t) {
    int tile_idx = t * BLOCK_SIZE + threadIdx.x;
    if (tile_idx < num_points)
      sh_points[threadIdx.x] = points[tile_idx];
    __syncthreads();

    if (idx < num_points) {
      int num_in_tile = min(BLOCK_SIZE, num_points - t * BLOCK_SIZE);
      for (int j = 0; j < num_in_tile; ++j) {
        if (euclidean_distance(my_point, sh_points[j]) <= eps)
          col_indices[start_offset + count++] = t * BLOCK_SIZE + j;
      }
    }
    __syncthreads();
  }
}

// KERNEL 2: Identifies core points (degree >= min_pts).
// A point is a core point if the number of neighbors (including itself)
// stored in the CSR row is at least min_pts. Result written to node_type[].
__global__ __launch_bounds__(BLOCK_SIZE) void identify_core_points_kernel(
    const int *__restrict__ row_ptr, int *__restrict__ node_type,
    int num_points, int min_pts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points)
    node_type[idx] = (row_ptr[idx + 1] - row_ptr[idx] >= min_pts) ? 1 : 0;
}

// KERNEL 3: BFS propagation — expands the current frontier.
// Each thread in the current frontier assigns the cluster ID to unvisited
// neighbors. Only core-point neighbors are added to the next frontier,
// preventing border points from further expanding the cluster.
__global__ __launch_bounds__(BLOCK_SIZE) void bfs_propagate_kernel(
    const int *__restrict__ row_ptr, const int *__restrict__ col_indices,
    int *__restrict__ cluster_ids, bool *__restrict__ current_frontier,
    bool *__restrict__ next_frontier, const int *__restrict__ node_type,
    int num_points, int current_cluster_id, bool *__restrict__ d_continue) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_points || !current_frontier[tid])
    return;

  // Visit all neighbors of this frontier node
  for (int e = row_ptr[tid]; e < row_ptr[tid + 1]; ++e) {
    int neighbor = col_indices[e];
    if (cluster_ids[neighbor] == 0) { // unvisited
      cluster_ids[neighbor] = current_cluster_id;
      // Only core points expand the frontier further
      if (node_type[neighbor] == 1) {
        next_frontier[neighbor] = true;
        *d_continue = true; // signal that BFS should continue
      }
    }
  }
  current_frontier[tid] = false; // remove current node from frontier
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

int main() {
  const std::string filename =
      "/content/drive/MyDrive/PDPProject/DATASET/GAIA_nearest_10000.csv";
  const float epsilon = 2.0f; // neighborhood radius for DBSCAN
  const int min_pts = 7;      // minimum neighbors to be a core point

  printf("G-DBSCAN v2 (improved)\nDataset: %s\nParams: Eps=%.2f, MinPts=%d\n",
         filename.c_str(), epsilon, min_pts);

  // 1. Load dataset from CSV into host memory
  std::vector<Point3D> h_points_vec;
  int num_points = load_dataset(filename, h_points_vec);
  if (num_points == 0) {
    fprintf(stderr, "Error: empty or missing dataset.\n");
    return 1;
  }

  // 2. Transfer points to device using Thrust (RAII, no manual cudaMalloc)
  thrust::device_vector<Point3D> d_points = h_points_vec;

  // 3. Auxiliary device structures (all managed by Thrust)
  thrust::device_vector<int> d_node_type(
      num_points); // 1 = core, 0 = border/noise
  thrust::device_vector<int> d_cluster_ids(num_points, 0); // 0 = unassigned
  thrust::device_vector<bool> d_frontier_a(num_points,
                                           false); // ping buffer for BFS
  thrust::device_vector<bool> d_frontier_b(num_points,
                                           false); // pong buffer for BFS
  thrust::device_vector<bool> d_continue_flag(1,
                                              false); // BFS continuation flag

  // Raw pointers used only for kernel calls (Thrust manages lifetime)
  const Point3D *rp_points = thrust::raw_pointer_cast(d_points.data());
  int *rp_node_type = thrust::raw_pointer_cast(d_node_type.data());
  int *rp_cluster_ids = thrust::raw_pointer_cast(d_cluster_ids.data());
  bool *rp_frontier_a = thrust::raw_pointer_cast(d_frontier_a.data());
  bool *rp_frontier_b = thrust::raw_pointer_cast(d_frontier_b.data());
  bool *rp_continue = thrust::raw_pointer_cast(d_continue_flag.data());

  // --- TIMING: measure total GPU execution time with CUDA events ---
  cudaEvent_t ev_start, ev_stop;
  CHECK_CUDA(cudaEventCreate(&ev_start));
  CHECK_CUDA(cudaEventCreate(&ev_stop));
  CHECK_CUDA(cudaEventRecord(ev_start));

  const int block = BLOCK_SIZE;
  const int grid = (num_points + block - 1) / block;

  // -------------------------------------------------------------------------
  // PHASE 1: Build CSR graph
  // -------------------------------------------------------------------------

  // 1a: Count neighbors (degrees) for each point
  thrust::device_vector<int> d_degrees(num_points);
  count_neighbors_kernel<<<grid, block>>>(
      rp_points, thrust::raw_pointer_cast(d_degrees.data()), num_points,
      epsilon);
  CHECK_CUDA(cudaDeviceSynchronize());

  // 1b: Exclusive prefix sum over degrees to produce row_ptr.
  // Minimal host transfer: only 2 scalar values are read from the device
  // to compute total_edges, avoiding a full copy.
  Graph g;
  g.row_ptr.resize(num_points + 1);
  thrust::exclusive_scan(d_degrees.begin(), d_degrees.end(), g.row_ptr.begin());
  int total_edges = g.row_ptr[num_points - 1] + d_degrees[num_points - 1];
  g.row_ptr[num_points] = total_edges; // sentinel: total number of edges

  printf("Phase 1b: Prefix sum done. Total edges: %d\n", total_edges);

  // 1c: Build col_indices using the computed row_ptr offsets
  g.col_indices.resize(total_edges);
  construct_adjacency_list<<<grid, block>>>(
      rp_points, thrust::raw_pointer_cast(g.row_ptr.data()),
      thrust::raw_pointer_cast(g.col_indices.data()), num_points, epsilon);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Raw pointers to graph arrays (stabilized after final resize)
  const int *rp_row_ptr = thrust::raw_pointer_cast(g.row_ptr.data());
  const int *rp_col_indices = thrust::raw_pointer_cast(g.col_indices.data());

  // -------------------------------------------------------------------------
  // PHASE 2: Identify core points
  // -------------------------------------------------------------------------
  identify_core_points_kernel<<<grid, block>>>(rp_row_ptr, rp_node_type,
                                               num_points, min_pts);
  CHECK_CUDA(cudaDeviceSynchronize());

  // -------------------------------------------------------------------------
  // PHASE 3: BFS Clustering
  // -------------------------------------------------------------------------

  // Copy node_type and cluster_ids to host; used for iterating seed selection
  // without launching a kernel for every candidate point
  std::vector<int> h_node_type(num_points);
  std::vector<int> h_cluster_ids(num_points, 0);
  thrust::copy(d_node_type.begin(), d_node_type.end(), h_node_type.begin());

  int current_cluster_id = 1; // cluster IDs start at 1 (0 = unassigned)

  for (int i = 0; i < num_points; ++i) {
    // Skip non-core or already-labeled points
    if (h_node_type[i] != 1 || h_cluster_ids[i] != 0)
      continue;

    // Seed a new cluster from this unvisited core point
    h_cluster_ids[i] = current_cluster_id;
    d_cluster_ids[i] =
        current_cluster_id; // implicit single-element copy to device
    d_frontier_a[i] = true; // mark seed as frontier entry point

    bool *ptr_in = rp_frontier_a;
    bool *ptr_out = rp_frontier_b;

    // Iterative BFS with double buffering (ping-pong between frontier_a and
    // frontier_b)
    while (true) {
      // Reset continuation flag before each BFS step
      d_continue_flag[0] = false;

      // Clear the OUTPUT buffer before writing new frontier nodes into it.
      // This is done before the swap so the kernel writes into a clean buffer.
      thrust::fill(thrust::device, ptr_out, ptr_out + num_points, false);

      bfs_propagate_kernel<<<grid, block>>>(
          rp_row_ptr, rp_col_indices, rp_cluster_ids, ptr_in, ptr_out,
          rp_node_type, num_points, current_cluster_id, rp_continue);
      CHECK_CUDA(cudaDeviceSynchronize());

      // Swap ping-pong buffers: old output becomes new input for next step
      std::swap(ptr_in, ptr_out);

      // The buffer that just became the new output (old input) will be cleared
      // at the start of the next iteration, so no extra reset is needed here.

      if (!d_continue_flag[0])
        break; // no new frontier nodes were added; BFS is complete
    }

    // Sync host cluster_ids with the device after finishing this cluster's BFS
    thrust::copy(d_cluster_ids.begin(), d_cluster_ids.end(),
                 h_cluster_ids.begin());
    current_cluster_id++;

    // Reset both frontier buffers before seeding the next cluster.
    // ptr_in may still hold stale data from the last BFS step, so both
    // buffers are cleared to be safe.
    thrust::fill(d_frontier_a.begin(), d_frontier_a.end(), false);
    thrust::fill(d_frontier_b.begin(), d_frontier_b.end(), false);
  }

  // --- TIMING END ---
  CHECK_CUDA(cudaEventRecord(ev_stop));
  CHECK_CUDA(cudaEventSynchronize(ev_stop));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start, ev_stop));

  printf("Execution Time: %.3f ms\n", ms);
  printf("Total Clusters Found: %d\n", current_cluster_id - 1);

  // Cleanup CUDA events (all device_vectors are freed automatically by Thrust)
  CHECK_CUDA(cudaEventDestroy(ev_start));
  CHECK_CUDA(cudaEventDestroy(ev_stop));

  return 0;
}