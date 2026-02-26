#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "/content/drive/MyDrive/PDPProject/SLINK/common.cuh"

// ================== STRUCTS ==================

// Represents a single edge in the Minimum Spanning Tree
struct MSTEdge {
  int u, v;   // endpoints (point indices)
  float dist; // Euclidean distance between u and v
};

// Parameters describing the 3D spatial grid used for neighbor acceleration
struct GridParams {
  float min_x, min_y, min_z;    // grid origin (lower-left-front corner)
  float cell_w, cell_h, cell_d; // cell dimensions along each axis
  int gx, gy, gz;               // number of cells per axis
  int total_cells;              // gx * gy * gz
};

// ================== HOST: BUILD 3D CELLULAR GRID ==================

// Builds a uniform 3D grid over the input point cloud.
// Points are sorted by cell ID (counting sort) into sorted_indices.
// cell_start[c] and cell_count[c] describe the slice of sorted_indices for cell
// c. Returns a GridParams struct describing the grid layout.
GridParams build_grid(const Point3D *points, int n, int *sorted_indices,
                      int *cell_start, int *cell_count) {
  // Compute axis-aligned bounding box of the point set
  float minx = FLT_MAX, miny = FLT_MAX, minz = FLT_MAX;
  float maxx = -FLT_MAX, maxy = -FLT_MAX, maxz = -FLT_MAX;
  for (int i = 0; i < n; i++) {
    if (points[i].x < minx)
      minx = points[i].x;
    if (points[i].y < miny)
      miny = points[i].y;
    if (points[i].z < minz)
      minz = points[i].z;
    if (points[i].x > maxx)
      maxx = points[i].x;
    if (points[i].y > maxy)
      maxy = points[i].y;
    if (points[i].z > maxz)
      maxz = points[i].z;
  }

  // Choose the grid resolution as the cube root of n, so each cell holds ~1
  // point on average
  int g = (int)cbrt((double)n);
  if (g < 1)
    g = 1;

  // Add a small epsilon to avoid points landing exactly on the upper boundary
  float eps = 1e-4f;
  float range_x = (maxx - minx) + eps;
  float range_y = (maxy - miny) + eps;
  float range_z = (maxz - minz) + eps;

  GridParams grid;
  grid.min_x =
      minx - eps * 0.5f; // shift origin slightly to avoid boundary edge cases
  grid.min_y = miny - eps * 0.5f;
  grid.min_z = minz - eps * 0.5f;
  grid.gx = g;
  grid.gy = g;
  grid.gz = g;
  grid.cell_w = range_x / g;
  grid.cell_h = range_y / g;
  grid.cell_d = range_z / g;
  grid.total_cells = g * g * g;

  // Compute cell ID for each point using a flattened 3D index: cx + cy*g +
  // cz*g*g
  std::vector<int> cell_ids(n);
  for (int i = 0; i < n; i++) {
    int cx = (int)((points[i].x - grid.min_x) / grid.cell_w);
    int cy = (int)((points[i].y - grid.min_y) / grid.cell_h);
    int cz = (int)((points[i].z - grid.min_z) / grid.cell_d);
    if (cx < 0)
      cx = 0;
    if (cx >= g)
      cx = g - 1;
    if (cy < 0)
      cy = 0;
    if (cy >= g)
      cy = g - 1;
    if (cz < 0)
      cz = 0;
    if (cz >= g)
      cz = g - 1;
    cell_ids[i] = cx + cy * g + cz * g * g;
  }

  // Counting sort by cell_id to fill sorted_indices
  for (int c = 0; c < grid.total_cells; c++)
    cell_count[c] = 0;
  for (int i = 0; i < n; i++)
    cell_count[cell_ids[i]]++;

  // Compute prefix sums to get starting positions in sorted_indices for each
  // cell
  int acc = 0;
  for (int c = 0; c < grid.total_cells; c++) {
    cell_start[c] = acc;
    acc += cell_count[c];
  }

  // Fill sorted_indices in cell order using a temporary offset array
  std::vector<int> temp_offset(cell_start, cell_start + grid.total_cells);
  for (int i = 0; i < n; i++) {
    int c = cell_ids[i];
    sorted_indices[temp_offset[c]] = i;
    temp_offset[c]++;
  }

  return grid;
}

// ================== KERNEL 1: FIND CLOSEST OUTGOING POINT ==================

// For each point i, finds the closest point j that belongs to a different
// component. The search expands outward shell by shell (Chebyshev shells)
// to exploit spatial locality from the grid and prune early.
// Results are written to out_target[i] (index) and out_dist[i] (distance).
__global__ __launch_bounds__(BLOCK_SIZE) void find_closest_outgoing(
    const Point3D *__restrict__ points, const int *__restrict__ sorted_indices,
    const int *__restrict__ cell_start, const int *__restrict__ cell_count,
    const int *__restrict__ comp_id, int n, GridParams grid,
    int *__restrict__ out_target, float *__restrict__ out_dist) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  Point3D p = points[i];
  int my_comp = comp_id[i];

  // Find the grid cell containing point i
  int cx0 = (int)((p.x - grid.min_x) / grid.cell_w);
  int cy0 = (int)((p.y - grid.min_y) / grid.cell_h);
  int cz0 = (int)((p.z - grid.min_z) / grid.cell_d);
  if (cx0 < 0)
    cx0 = 0;
  if (cx0 >= grid.gx)
    cx0 = grid.gx - 1;
  if (cy0 < 0)
    cy0 = 0;
  if (cy0 >= grid.gy)
    cy0 = grid.gy - 1;
  if (cz0 < 0)
    cz0 = 0;
  if (cz0 >= grid.gz)
    cz0 = grid.gz - 1;

  float best_dist = FLT_MAX;
  int best_target = -1;

  // Maximum shell radius to search (bounded by the longest grid dimension)
  int max_r = grid.gx;
  if (grid.gy > max_r)
    max_r = grid.gy;
  if (grid.gz > max_r)
    max_r = grid.gz;

  // Expand search radius r, visiting only the surface cells of each shell
  for (int r = 0; r < max_r; r++) {
    // Early termination: if the minimum possible distance to the next shell
    // is already >= the current best, no closer point can be found
    if (r >= 2 && best_target >= 0) {
      float shell_min_dx = (float)(r - 1) * grid.cell_w;
      float shell_min_dy = (float)(r - 1) * grid.cell_h;
      float shell_min_dz = (float)(r - 1) * grid.cell_d;
      float shell_min = shell_min_dx;
      if (shell_min_dy < shell_min)
        shell_min = shell_min_dy;
      if (shell_min_dz < shell_min)
        shell_min = shell_min_dz;
      if (shell_min >= best_dist)
        break;
    }

    // Clamp search range to grid bounds
    int lo_x = max(0, cx0 - r);
    int hi_x = min(grid.gx - 1, cx0 + r);
    int lo_y = max(0, cy0 - r);
    int hi_y = min(grid.gy - 1, cy0 + r);
    int lo_z = max(0, cz0 - r);
    int hi_z = min(grid.gz - 1, cz0 + r);

    for (int cz = lo_z; cz <= hi_z; cz++) {
      for (int cy = lo_y; cy <= hi_y; cy++) {
        for (int cx = lo_x; cx <= hi_x; cx++) {
          // Only visit cells on the shell surface (Chebyshev distance == r)
          if (r > 0) {
            int dx = abs(cx - cx0);
            int dy = abs(cy - cy0);
            int dz = abs(cz - cz0);
            int cheb = max(max(dx, dy), dz);
            if (cheb != r)
              continue;
          }

          int cell_id = cx + cy * grid.gx + cz * grid.gx * grid.gy;
          int start = cell_start[cell_id];
          int cnt = cell_count[cell_id];

          // Check every point in this cell
          for (int k = 0; k < cnt; k++) {
            int j = sorted_indices[start + k];
            if (j == i)
              continue; // skip self
            if (comp_id[j] == my_comp)
              continue; // skip same component

            float d = euclidean_distance(p, points[j]);

            // Update best, with tie-breaking by smaller index for determinism
            if (d < best_dist || (d == best_dist && j < best_target)) {
              best_dist = d;
              best_target = j;
            }
          }
        }
      }
    }
  }
  out_target[i] = best_target;
  out_dist[i] = best_dist;
}

// ================== KERNEL 2: FIND COMPONENT MIN EDGE ==================

// Encodes (dist, src) into a single 64-bit integer for atomic minimum
// operations. The distance bits occupy the high 32 bits, so atomicMin naturally
// selects the edge with the smallest distance (and smallest source index as
// tie-breaker).
__device__ unsigned long long encode_edge(float dist, int src) {
  unsigned int dist_bits = __float_as_uint(dist);
  return ((unsigned long long)dist_bits << 32) | (unsigned int)src;
}

// For each point i that has a valid outgoing edge, atomically updates the
// minimum encoded edge for its component in comp_best[].
__global__ __launch_bounds__(BLOCK_SIZE) void find_component_min_edge(
    const int *__restrict__ comp_id, const int *__restrict__ out_target,
    const float *__restrict__ out_dist,
    unsigned long long *__restrict__ comp_best, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || out_target[i] < 0)
    return;

  int comp = comp_id[i];
  unsigned long long val = encode_edge(out_dist[i], i);
  atomicMin(&comp_best[comp],
            val); // keep the smallest (dist, src) per component
}

// ================== KERNEL 3: COLLECT MST EDGES ==================

// For each component, the thread whose index matches the winner (lowest encoded
// edge) adds the corresponding edge to the MST edge list. Mutual edges between
// two components are deduplicated: only the component with the smaller ID adds
// the edge.
__global__ __launch_bounds__(BLOCK_SIZE) void collect_mst_edges(
    const int *__restrict__ comp_id, const int *__restrict__ out_target,
    const float *__restrict__ out_dist,
    const unsigned long long *__restrict__ comp_best,
    MSTEdge *__restrict__ mst_edges, int *__restrict__ mst_count, int max_edges,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int comp = comp_id[i];
  unsigned long long best = comp_best[comp];
  if (best == ULLONG_MAX)
    return; // component has no outgoing edge

  // Only the winning thread (the one selected by atomicMin) adds the edge
  int winner = (int)(best & 0xFFFFFFFF);
  if (i != winner)
    return;

  int target = out_target[i];
  if (target < 0)
    return;
  int target_comp = comp_id[target];
  if (target_comp == comp)
    return; // edge is within the same component (should not happen)

  // Deduplicate mutual edges: if both components selected the same cross-edge,
  // only the component with the smaller ID writes it to avoid duplicates
  if (comp > target_comp) {
    unsigned long long tb = comp_best[target_comp];
    if (tb != ULLONG_MAX) {
      int tw = (int)(tb & 0xFFFFFFFF);
      if (tw >= 0 && tw < n) {
        int tt = out_target[tw];
        if (tt >= 0 && comp_id[tt] == comp) {
          return; // mutual: the smaller comp already adds this edge
        }
      }
    }
  }

  // Bounds check before writing to avoid overflow of the MST edge buffer
  int idx = atomicAdd(mst_count, 1);
  if (idx < max_edges) {
    mst_edges[idx].u = i;
    mst_edges[idx].v = target;
    mst_edges[idx].dist = out_dist[i];
  }
}

// ================== KERNEL 4a: MERGE COMPONENTS ==================

// Merges pairs of components connected by their chosen minimum outgoing edges.
// Only root nodes (comp_id[i] == i) perform the merge to avoid race conditions.
// The merge assigns the smaller component ID to the larger one via atomicCAS.
__global__ __launch_bounds__(BLOCK_SIZE) void merge_components(
    int *__restrict__ comp_id, const int *__restrict__ out_target,
    const unsigned long long *__restrict__ comp_best, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int comp = comp_id[i];
  if (comp != i)
    return; // only roots process merges

  unsigned long long best = comp_best[comp];
  if (best == ULLONG_MAX)
    return; // no outgoing edge for this component

  int winner = (int)(best & 0xFFFFFFFF);
  int target = out_target[winner];
  if (target < 0)
    return;

  int target_comp = comp_id[target];
  if (target_comp == comp)
    return;

  // Merge: point the larger-ID root to the smaller-ID root
  int lo = min(comp, target_comp);
  int hi = max(comp, target_comp);

  atomicCAS(&comp_id[hi], hi, lo); // only succeeds if hi is still a root
}

// ================== KERNEL 4b: POINTER JUMPING ==================

// Flattens the component ID tree by having each node point directly to its
// grandparent. Repeated until no more changes occur (convergence).
// d_changed is set to 1 if any node updated its comp_id in this pass.
__global__ __launch_bounds__(BLOCK_SIZE) void pointer_jumping(
    int *__restrict__ comp_id, int n, int *__restrict__ changed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int comp = comp_id[i];
  int parent = comp_id[comp]; // grandparent of i

  if (parent != comp) {
    comp_id[i] = parent; // path compression: skip one level
    *changed = 1;
  }
}

// ================== HOST: UNION-FIND FOR POST-PROCESSING ==================

// CPU-side Union-Find with path compression and union-by-rank.
// Used during post-processing to deduplicate and filter raw MST edges.
struct UnionFind {
  std::vector<int> parent;
  std::vector<int> rank_;

  void init(int n) {
    parent.resize(n);
    rank_.assign(n, 0);
    for (int i = 0; i < n; i++)
      parent[i] = i;
  }

  // Find root with path halving (iterative path compression)
  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]]; // path halving
      x = parent[x];
    }
    return x;
  }

  // Union by rank; returns false if a and b are already in the same set
  bool unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return false;
    if (rank_[a] < rank_[b])
      std::swap(a, b);
    parent[b] = a;
    if (rank_[a] == rank_[b])
      rank_[a]++;
    return true;
  }
};

// ================== MAIN ==================

int main() {
  auto total_start = std::chrono::high_resolution_clock::now();

  // Edge distance threshold for final flat clustering (cut the MST at this
  // length)
  const float cut_threshold = 2.0f;
  std::string GAIA_CARTESIAN_FILENAME = "GAIA_nearest_200000.csv";
  std::string filename =
      "/content/drive/MyDrive/PDPProject/" + GAIA_CARTESIAN_FILENAME;

  // Load the 3D point cloud from CSV
  std::vector<Point3D> h_points;
  int n = load_dataset(filename, h_points);
  if (n == 0)
    return 1;

  printf("\n=== BORUVKA-ELIAS EMST (CUDA) ===\n");
  printf("Dataset: %s\n", filename.c_str());
  printf("Loaded points: %d\n", n);

  auto phase_start = std::chrono::high_resolution_clock::now();

  // Grid resolution: cube root of n, so expected ~1 point per cell
  int g = std::max(1, (int)cbrt((double)n));
  int total_cells = g * g * g;

  std::vector<int> h_sorted_indices(n);
  std::vector<int> h_cell_start(total_cells);
  std::vector<int> h_cell_count(total_cells);

  // Build the spatial grid on the host; results are stored in h_sorted_indices,
  // h_cell_start, and h_cell_count
  GridParams grid = build_grid(h_points.data(), n, h_sorted_indices.data(),
                               h_cell_start.data(), h_cell_count.data());

  auto phase_end = std::chrono::high_resolution_clock::now();
  double t_grid =
      std::chrono::duration<double>(phase_end - phase_start).count();

  printf("Grid: %d x %d x %d = %d cells\n", grid.gx, grid.gy, grid.gz,
         grid.total_cells);

  phase_start = std::chrono::high_resolution_clock::now();

  // Upper bound on the number of MST edges that can be collected across all
  // iterations
  int max_edges = 3 * n;

  // Transfer all host data to device using Thrust (RAII, no manual
  // cudaMalloc/cudaFree)
  thrust::device_vector<Point3D> d_points = h_points;
  thrust::device_vector<int> d_sorted_indices = h_sorted_indices;
  thrust::device_vector<int> d_cell_start = h_cell_start;
  thrust::device_vector<int> d_cell_count = h_cell_count;

  thrust::device_vector<int> d_comp_id(n); // component ID for each point
  thrust::device_vector<int> d_out_target(
      n); // closest out-of-component neighbor for each point
  thrust::device_vector<float> d_out_dist(n); // distance to that neighbor
  thrust::device_vector<unsigned long long> d_comp_best(
      n); // encoded best edge per component
  thrust::device_vector<MSTEdge> d_mst_edges(max_edges); // collected MST edges
  thrust::device_vector<int> d_mst_count(
      1, 0); // number of MST edges collected so far
  thrust::device_vector<int> d_changed(
      1, 0); // flag for pointer jumping convergence

  // Raw pointers for kernels (Thrust manages the underlying memory)
  Point3D *ptr_points = thrust::raw_pointer_cast(d_points.data());
  int *ptr_sorted_indices = thrust::raw_pointer_cast(d_sorted_indices.data());
  int *ptr_cell_start = thrust::raw_pointer_cast(d_cell_start.data());
  int *ptr_cell_count = thrust::raw_pointer_cast(d_cell_count.data());
  int *ptr_comp_id = thrust::raw_pointer_cast(d_comp_id.data());
  int *ptr_out_target = thrust::raw_pointer_cast(d_out_target.data());
  float *ptr_out_dist = thrust::raw_pointer_cast(d_out_dist.data());
  unsigned long long *ptr_comp_best =
      thrust::raw_pointer_cast(d_comp_best.data());
  MSTEdge *ptr_mst_edges = thrust::raw_pointer_cast(d_mst_edges.data());
  int *ptr_mst_count = thrust::raw_pointer_cast(d_mst_count.data());
  int *ptr_changed = thrust::raw_pointer_cast(d_changed.data());

  // Initialize comp_id[i] = i: each point starts in its own singleton component
  thrust::sequence(d_comp_id.begin(), d_comp_id.end());

  int block = BLOCK_SIZE;
  int nblocks = (n + block - 1) / block;

  phase_end = std::chrono::high_resolution_clock::now();
  double t_alloc =
      std::chrono::duration<double>(phase_end - phase_start).count();

  // --- PHASE 4: BORUVKA ITERATIONS ---
  // Each iteration halves (at minimum) the number of components by merging
  // pairs connected by their shortest cross-component edges.
  phase_start = std::chrono::high_resolution_clock::now();

  int iteration = 0;
  int num_components = n; // initially one component per point

  while (num_components > 1) {
    iteration++;

    // Step 1: For each point, find the closest point in a different component
    thrust::fill(d_out_target.begin(), d_out_target.end(), -1);
    thrust::fill(d_out_dist.begin(), d_out_dist.end(), FLT_MAX);

    find_closest_outgoing<<<nblocks, block>>>(
        ptr_points, ptr_sorted_indices, ptr_cell_start, ptr_cell_count,
        ptr_comp_id, n, grid, ptr_out_target, ptr_out_dist);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 2: For each component, find the shortest outgoing edge
    thrust::fill(d_comp_best.begin(), d_comp_best.end(), ULLONG_MAX);

    find_component_min_edge<<<nblocks, block>>>(ptr_comp_id, ptr_out_target,
                                                ptr_out_dist, ptr_comp_best, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 3: Add selected minimum edges to the MST edge list
    collect_mst_edges<<<nblocks, block>>>(
        ptr_comp_id, ptr_out_target, ptr_out_dist, ptr_comp_best, ptr_mst_edges,
        ptr_mst_count, max_edges, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 4a: Merge components by updating comp_id for each root
    merge_components<<<nblocks, block>>>(ptr_comp_id, ptr_out_target,
                                         ptr_comp_best, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 4b: Flatten the component ID tree via repeated pointer jumping
    while (true) {
      d_changed[0] = 0; // reset convergence flag
      pointer_jumping<<<nblocks, block>>>(ptr_comp_id, n, ptr_changed);
      CHECK_CUDA(cudaDeviceSynchronize());
      if (d_changed[0] == 0)
        break; // no updates: tree is fully flattened
    }

    // Count distinct component IDs to track convergence
    thrust::device_vector<int> comp_id_copy = d_comp_id;
    thrust::sort(comp_id_copy.begin(), comp_id_copy.end());
    auto new_end = thrust::unique(comp_id_copy.begin(), comp_id_copy.end());
    num_components = new_end - comp_id_copy.begin();

    int raw_edge_count = d_mst_count[0];

    printf("   Iteration %d: components = %d, raw MST edges = %d\n", iteration,
           num_components, raw_edge_count);

    if (iteration > 100) {
      printf("   WARNING: max iterations reached\n");
      break;
    }
  }

  phase_end = std::chrono::high_resolution_clock::now();
  double t_boruvka =
      std::chrono::duration<double>(phase_end - phase_start).count();

  // --- PHASE 5: POST-PROCESS MST EDGES (remove redundant) ---
  // Boruvka may produce duplicate or redundant edges; filter them using
  // Kruskal's algorithm on the host with a Union-Find structure.
  phase_start = std::chrono::high_resolution_clock::now();

  int raw_edge_count = std::min((int)d_mst_count[0], max_edges);

  // Copy raw edge list from device to host
  std::vector<MSTEdge> h_raw_edges(raw_edge_count);
  thrust::copy(d_mst_edges.begin(), d_mst_edges.begin() + raw_edge_count,
               h_raw_edges.begin());

  // Sort edges by distance (ascending) for Kruskal's greedy selection
  std::sort(h_raw_edges.begin(), h_raw_edges.end(),
            [](const MSTEdge &a, const MSTEdge &b) { return a.dist < b.dist; });

  // Kruskal filter: keep only edges that connect two different components
  UnionFind uf;
  uf.init(n);

  std::vector<MSTEdge> h_mst_edges;
  h_mst_edges.reserve(n - 1);

  for (const auto &edge : h_raw_edges) {
    if (h_mst_edges.size() >= n - 1)
      break; // MST is complete
    if (uf.unite(edge.u, edge.v)) {
      h_mst_edges.push_back(edge); // edge connects two components: add to MST
    }
  }

  // Derive flat clustering by counting edges longer than cut_threshold that
  // would be removed (each removal splits one cluster into two)
  int num_clusters = n;
  for (const auto &edge : h_mst_edges) {
    if (edge.dist <= cut_threshold)
      num_clusters--; // merging: reduces cluster count by 1
  }

  phase_end = std::chrono::high_resolution_clock::now();
  double t_cluster =
      std::chrono::duration<double>(phase_end - phase_start).count();

  // --- OUTPUT ---
  auto total_end = std::chrono::high_resolution_clock::now();
  double t_total =
      std::chrono::duration<double>(total_end - total_start).count();

  printf("\n=== RESULTS ===\n");
  printf("Loaded points             : %d\n", n);
  printf("Raw edges collected       : %d\n", raw_edge_count);
  printf("MST edges (after filter)  : %ld / %d\n", h_mst_edges.size(), n - 1);
  printf("Boruvka iterations        : %d\n", iteration);
  printf("Clusters (threshold %.1f) : %d\n", cut_threshold, num_clusters);
  printf("\nExecution time:\n");
  printf("   Grid construction  : %.4f s\n", t_grid);
  printf("   Device allocation  : %.4f s\n", t_alloc);
  printf("   Boruvka iterations : %.4f s\n", t_boruvka);
  printf("   Post-process       : %.4f s\n", t_cluster);
  printf("   TOTAL              : %.4f s\n", t_total);

  // Save the MST edge list as a linkage CSV for downstream dendrogram analysis
  std::ofstream out("linkage.csv");
  out << "idx1,idx2,dist\n";
  for (const auto &edge : h_mst_edges) {
    out << edge.u << "," << edge.v << "," << edge.dist << "\n";
  }
  printf("\nSaved linkage.csv (%ld edges)\n", h_mst_edges.size());

  // --- CLEANUP ---
  // No explicit cudaFree required thanks to thrust::device_vector

  return 0;
}
