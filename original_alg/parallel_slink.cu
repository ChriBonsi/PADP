#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <climits>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t _err = (call); \
  if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(_err)); \
    exit(1); \
  } \
} while(0)

// ================== STRUCTS ==================

struct Point3D {
    float x, y, z;
};

struct MSTEdge {
    int u, v;
    float dist;
};

struct GridParams {
    float min_x, min_y, min_z;
    float cell_w, cell_h, cell_d;
    int gx, gy, gz;
    int total_cells;
};

// ================== HOST: LOAD DATASET ==================

int load_dataset(const std::string& filename, std::vector<Point3D>& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename.c_str());
        return 0;
    }
    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        Point3D p = {0, 0, 0};
        std::getline(ss, cell, ','); // skip source_id
        std::getline(ss, cell, ','); try { p.x = std::stof(cell); } catch (...) {}
        std::getline(ss, cell, ','); try { p.y = std::stof(cell); } catch (...) {}
        std::getline(ss, cell, ','); try { p.z = std::stof(cell); } catch (...) {}
        points.push_back(p);
    }
    return (int)points.size();
}

// ================== HOST: BUILD 3D CELLULAR GRID ==================

GridParams build_grid(const Point3D* points, int n,
                      int* sorted_indices,
                      int* cell_start,
                      int* cell_count) {
    float minx = FLT_MAX, miny = FLT_MAX, minz = FLT_MAX;
    float maxx = -FLT_MAX, maxy = -FLT_MAX, maxz = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (points[i].x < minx) minx = points[i].x;
        if (points[i].y < miny) miny = points[i].y;
        if (points[i].z < minz) minz = points[i].z;
        if (points[i].x > maxx) maxx = points[i].x;
        if (points[i].y > maxy) maxy = points[i].y;
        if (points[i].z > maxz) maxz = points[i].z;
    }

    int g = (int)cbrt((double)n);
    if (g < 1) g = 1;

    float eps = 1e-4f;
    float range_x = (maxx - minx) + eps;
    float range_y = (maxy - miny) + eps;
    float range_z = (maxz - minz) + eps;

    GridParams grid;
    grid.min_x = minx - eps * 0.5f;
    grid.min_y = miny - eps * 0.5f;
    grid.min_z = minz - eps * 0.5f;
    grid.gx = g; grid.gy = g; grid.gz = g;
    grid.cell_w = range_x / g;
    grid.cell_h = range_y / g;
    grid.cell_d = range_z / g;
    grid.total_cells = g * g * g;

    // Compute cell ID for each point
    int* cell_ids = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int cx = (int)((points[i].x - grid.min_x) / grid.cell_w);
        int cy = (int)((points[i].y - grid.min_y) / grid.cell_h);
        int cz = (int)((points[i].z - grid.min_z) / grid.cell_d);
        if (cx < 0) cx = 0; if (cx >= g) cx = g - 1;
        if (cy < 0) cy = 0; if (cy >= g) cy = g - 1;
        if (cz < 0) cz = 0; if (cz >= g) cz = g - 1;
        cell_ids[i] = cx + cy * g + cz * g * g;
    }

    // Counting sort by cell_id
    for (int c = 0; c < grid.total_cells; c++) cell_count[c] = 0;
    for (int i = 0; i < n; i++) cell_count[cell_ids[i]]++;

    int acc = 0;
    for (int c = 0; c < grid.total_cells; c++) {
        cell_start[c] = acc;
        acc += cell_count[c];
    }

    int* temp_offset = (int*)malloc(grid.total_cells * sizeof(int));
    for (int c = 0; c < grid.total_cells; c++) temp_offset[c] = cell_start[c];
    for (int i = 0; i < n; i++) {
        int c = cell_ids[i];
        sorted_indices[temp_offset[c]] = i;
        temp_offset[c]++;
    }

    free(temp_offset);
    free(cell_ids);
    return grid;
}

// ================== INIT KERNELS ==================

__global__ void init_int(int* arr, int val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

__global__ void init_float(float* arr, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

__global__ void init_ull(unsigned long long* arr, unsigned long long val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

__global__ void init_sequence(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

// ================== KERNEL 1: FIND CLOSEST OUTGOING POINT ==================

__global__ void find_closest_outgoing(
    const Point3D* __restrict__ points,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_count,
    const int* __restrict__ comp_id,
    int n,
    GridParams grid,
    int* __restrict__ out_target,
    float* __restrict__ out_dist
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = points[i].x, py = points[i].y, pz = points[i].z;
    int my_comp = comp_id[i];

    int cx0 = (int)((px - grid.min_x) / grid.cell_w);
    int cy0 = (int)((py - grid.min_y) / grid.cell_h);
    int cz0 = (int)((pz - grid.min_z) / grid.cell_d);
    if (cx0 < 0) cx0 = 0; if (cx0 >= grid.gx) cx0 = grid.gx - 1;
    if (cy0 < 0) cy0 = 0; if (cy0 >= grid.gy) cy0 = grid.gy - 1;
    if (cz0 < 0) cz0 = 0; if (cz0 >= grid.gz) cz0 = grid.gz - 1;

    float best_dist = FLT_MAX;
    int best_target = -1;

    int max_r = grid.gx;
    if (grid.gy > max_r) max_r = grid.gy;
    if (grid.gz > max_r) max_r = grid.gz;

    for (int r = 0; r < max_r; r++) {
        // Termination: closest possible distance at shell r exceeds best
        if (r >= 2 && best_target >= 0) {
            float shell_min_dx = (float)(r - 1) * grid.cell_w;
            float shell_min_dy = (float)(r - 1) * grid.cell_h;
            float shell_min_dz = (float)(r - 1) * grid.cell_d;
            float shell_min = shell_min_dx;
            if (shell_min_dy < shell_min) shell_min = shell_min_dy;
            if (shell_min_dz < shell_min) shell_min = shell_min_dz;
            if (shell_min >= best_dist) break;
        }

        int lo_x = cx0 - r; if (lo_x < 0) lo_x = 0;
        int hi_x = cx0 + r; if (hi_x >= grid.gx) hi_x = grid.gx - 1;
        int lo_y = cy0 - r; if (lo_y < 0) lo_y = 0;
        int hi_y = cy0 + r; if (hi_y >= grid.gy) hi_y = grid.gy - 1;
        int lo_z = cz0 - r; if (lo_z < 0) lo_z = 0;
        int hi_z = cz0 + r; if (hi_z >= grid.gz) hi_z = grid.gz - 1;

        for (int cz = lo_z; cz <= hi_z; cz++) {
            for (int cy = lo_y; cy <= hi_y; cy++) {
                for (int cx = lo_x; cx <= hi_x; cx++) {
                    // Only shell surface (Chebyshev distance == r)
                    if (r > 0) {
                        int dx = cx - cx0; if (dx < 0) dx = -dx;
                        int dy = cy - cy0; if (dy < 0) dy = -dy;
                        int dz = cz - cz0; if (dz < 0) dz = -dz;
                        int cheb = dx; if (dy > cheb) cheb = dy; if (dz > cheb) cheb = dz;
                        if (cheb != r) continue;
                    }

                    int cell_id = cx + cy * grid.gx + cz * grid.gx * grid.gy;
                    int start = cell_start[cell_id];
                    int cnt   = cell_count[cell_id];

                    for (int k = 0; k < cnt; k++) {
                        int j = sorted_indices[start + k];
                        if (j == i) continue;
                        if (comp_id[j] == my_comp) continue;

                        float ddx = px - points[j].x;
                        float ddy = py - points[j].y;
                        float ddz = pz - points[j].z;
                        float d = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);

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
    out_dist[i]   = best_dist;
}

// ================== KERNEL 2: FIND COMPONENT MIN EDGE ==================

__device__ unsigned long long encode_edge(float dist, int src) {
    unsigned int dist_bits = __float_as_uint(dist);
    return ((unsigned long long)dist_bits << 32) | (unsigned int)src;
}

__global__ void find_component_min_edge(
    const int* __restrict__ comp_id,
    const int* __restrict__ out_target,
    const float* __restrict__ out_dist,
    unsigned long long* __restrict__ comp_best,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (out_target[i] < 0) return;

    int comp = comp_id[i];
    unsigned long long val = encode_edge(out_dist[i], i);
    atomicMin(&comp_best[comp], val);
}

// ================== KERNEL 3: COLLECT MST EDGES ==================
// Each component's winner adds one MST edge.
// Dedup mutual edges: only the smaller comp_id side adds.
// Bounds-checked to avoid buffer overflow.

__global__ void collect_mst_edges(
    const int* __restrict__ comp_id,
    const int* __restrict__ out_target,
    const float* __restrict__ out_dist,
    const unsigned long long* __restrict__ comp_best,
    MSTEdge* __restrict__ mst_edges,
    int* __restrict__ mst_count,
    int max_edges,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int comp = comp_id[i];
    unsigned long long best = comp_best[comp];
    if (best == ULLONG_MAX) return;

    int winner = (int)(best & 0xFFFFFFFF);
    if (i != winner) return;

    int target = out_target[i];
    if (target < 0) return;
    int target_comp = comp_id[target];
    if (target_comp == comp) return;

    // Dedup mutual edges
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

    // Bounds check before writing
    int idx = atomicAdd(mst_count, 1);
    if (idx < max_edges) {
        mst_edges[idx].u = i;
        mst_edges[idx].v = target;
        mst_edges[idx].dist = out_dist[i];
    }
}

// ================== KERNEL 4a: MERGE COMPONENTS ==================

__global__ void merge_components(
    int* __restrict__ comp_id,
    const int* __restrict__ out_target,
    const unsigned long long* __restrict__ comp_best,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int comp = comp_id[i];
    if (comp != i) return; // only roots

    unsigned long long best = comp_best[comp];
    if (best == ULLONG_MAX) return;

    int winner = (int)(best & 0xFFFFFFFF);
    int target = out_target[winner];
    if (target < 0) return;

    int target_comp = comp_id[target];
    if (target_comp == comp) return;

    int lo = comp < target_comp ? comp : target_comp;
    int hi = comp < target_comp ? target_comp : comp;

    atomicCAS(&comp_id[hi], hi, lo);
}

// ================== KERNEL 4b: POINTER JUMPING ==================

__global__ void pointer_jumping(int* __restrict__ comp_id, int n, int* __restrict__ changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int comp = comp_id[i];
    int parent = comp_id[comp];

    if (parent != comp) {
        comp_id[i] = parent;
        *changed = 1;
    }
}

// ================== HOST: UNION-FIND FOR POST-PROCESSING ==================

struct UnionFind {
    int* parent;
    int* rank_;
    int n;

    void init(int nn) {
        n = nn;
        parent = (int*)malloc(n * sizeof(int));
        rank_  = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) { parent[i] = i; rank_[i] = 0; }
    }

    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }

    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) { int t = a; a = b; b = t; }
        parent[b] = a;
        if (rank_[a] == rank_[b]) rank_[a]++;
        return true;
    }

    void destroy() { free(parent); free(rank_); }
};

// ================== MAIN ==================

int main() {
    auto total_start = std::chrono::high_resolution_clock::now();

    // --- PARAMETERS ---
    const float cut_threshold = 2.0f;

    // --- PHASE 1: LOAD DATA ---
    // std::string GAIA_CARTESIAN_FILENAME = "GAIA_DR3_Cartesian_Heliocentric.csv";
    std::string GAIA_CARTESIAN_FILENAME = "GAIA_nearest_10000.csv";
    // std::string GAIA_CARTESIAN_FILENAME = "GAIA_nearest_200000.csv";
    std::string filename = "/content/drive/MyDrive/PDPProject/" + GAIA_CARTESIAN_FILENAME;

    std::vector<Point3D> h_points;
    int n = load_dataset(filename, h_points);
    if (n == 0) { fprintf(stderr, "Failed to load dataset.\n"); return 1; }

    printf("\n=== BORUVKA-ELIAS EMST (CUDA) ===\n");
    printf("Dataset: %s\n", filename.c_str());
    printf("Loaded points: %d\n", n);

    // --- PHASE 2: BUILD 3D CELLULAR GRID ---
    auto phase_start = std::chrono::high_resolution_clock::now();

    int g = (int)cbrt((double)n);
    if (g < 1) g = 1;
    int total_cells = g * g * g;

    int* h_sorted_indices = (int*)malloc(n * sizeof(int));
    int* h_cell_start     = (int*)malloc(total_cells * sizeof(int));
    int* h_cell_count     = (int*)malloc(total_cells * sizeof(int));

    GridParams grid = build_grid(h_points.data(), n, h_sorted_indices,
                                 h_cell_start, h_cell_count);

    auto phase_end = std::chrono::high_resolution_clock::now();
    double t_grid = std::chrono::duration<double>(phase_end - phase_start).count();

    printf("Grid: %d x %d x %d = %d cells\n", grid.gx, grid.gy, grid.gz, grid.total_cells);

    // --- PHASE 3: ALLOCATE DEVICE MEMORY ---
    phase_start = std::chrono::high_resolution_clock::now();

    // Oversized MST edge buffer: Boruvka can produce redundant edges
    // across iterations, so we allow up to 3*n edges and filter later
    int max_edges = 3 * n;

    Point3D* d_points;
    int* d_sorted_indices;
    int* d_cell_start;
    int* d_cell_count;
    int* d_comp_id;
    int* d_out_target;
    float* d_out_dist;
    unsigned long long* d_comp_best;
    MSTEdge* d_mst_edges;
    int* d_mst_count;
    int* d_changed;

    CHECK_CUDA(cudaMalloc(&d_points, n * sizeof(Point3D)));
    CHECK_CUDA(cudaMalloc(&d_sorted_indices, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cell_start, total_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cell_count, total_cells * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_comp_id, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out_target, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out_dist, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_comp_best, n * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_mst_edges, max_edges * sizeof(MSTEdge)));
    CHECK_CUDA(cudaMalloc(&d_mst_count, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_changed, sizeof(int)));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_points, h_points.data(), n * sizeof(Point3D), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sorted_indices, h_sorted_indices, n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cell_start, h_cell_start, total_cells * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cell_count, h_cell_count, total_cells * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_mst_count, 0, sizeof(int)));

    // Initialize comp_id[i] = i
    int block = 256;
    int nblocks = (n + block - 1) / block;
    init_sequence<<<nblocks, block>>>(d_comp_id, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    phase_end = std::chrono::high_resolution_clock::now();
    double t_alloc = std::chrono::duration<double>(phase_end - phase_start).count();

    // --- PHASE 4: BORUVKA ITERATIONS ---
    phase_start = std::chrono::high_resolution_clock::now();

    int iteration = 0;
    int num_components = n;
    int* h_comp_ids = (int*)malloc(n * sizeof(int));

    while (num_components > 1) {
        iteration++;

        // Step 1: Init + find closest outgoing point for each vertex
        init_int<<<nblocks, block>>>(d_out_target, -1, n);
        init_float<<<nblocks, block>>>(d_out_dist, FLT_MAX, n);
        CHECK_CUDA(cudaDeviceSynchronize());

        find_closest_outgoing<<<nblocks, block>>>(
            d_points, d_sorted_indices, d_cell_start, d_cell_count,
            d_comp_id, n, grid, d_out_target, d_out_dist
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // Step 2: Find each component's shortest outgoing edge
        init_ull<<<nblocks, block>>>(d_comp_best, ULLONG_MAX, n);
        CHECK_CUDA(cudaDeviceSynchronize());

        find_component_min_edge<<<nblocks, block>>>(
            d_comp_id, d_out_target, d_out_dist, d_comp_best, n
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // Step 3: Collect MST edges (bounds-checked)
        collect_mst_edges<<<nblocks, block>>>(
            d_comp_id, d_out_target, d_out_dist, d_comp_best,
            d_mst_edges, d_mst_count, max_edges, n
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // Step 4: Merge components
        merge_components<<<nblocks, block>>>(
            d_comp_id, d_out_target, d_comp_best, n
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // Pointer jumping until convergence
        int h_changed = 1;
        while (h_changed) {
            h_changed = 0;
            CHECK_CUDA(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
            pointer_jumping<<<nblocks, block>>>(d_comp_id, n, d_changed);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        }

        // Count remaining components
        CHECK_CUDA(cudaMemcpy(h_comp_ids, d_comp_id, n * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> roots(h_comp_ids, h_comp_ids + n);
        std::sort(roots.begin(), roots.end());
        num_components = (int)(std::unique(roots.begin(), roots.end()) - roots.begin());

        int raw_edge_count;
        CHECK_CUDA(cudaMemcpy(&raw_edge_count, d_mst_count, sizeof(int), cudaMemcpyDeviceToHost));

        printf("   Iteration %d: components = %d, raw MST edges = %d\n",
               iteration, num_components, raw_edge_count);

        if (iteration > 100) {
            printf("   WARNING: max iterations reached\n");
            break;
        }
    }

    phase_end = std::chrono::high_resolution_clock::now();
    double t_boruvka = std::chrono::duration<double>(phase_end - phase_start).count();

    // --- PHASE 5: POST-PROCESS MST EDGES (remove redundant) ---
    phase_start = std::chrono::high_resolution_clock::now();

    int raw_edge_count;
    CHECK_CUDA(cudaMemcpy(&raw_edge_count, d_mst_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (raw_edge_count > max_edges) raw_edge_count = max_edges;

    MSTEdge* h_raw_edges = (MSTEdge*)malloc(raw_edge_count * sizeof(MSTEdge));
    CHECK_CUDA(cudaMemcpy(h_raw_edges, d_mst_edges,
                          raw_edge_count * sizeof(MSTEdge), cudaMemcpyDeviceToHost));

    // Sort by distance
    std::sort(h_raw_edges, h_raw_edges + raw_edge_count,
              [](const MSTEdge& a, const MSTEdge& b) { return a.dist < b.dist; });

    // Kruskal filter: keep only edges that actually merge two components
    UnionFind uf;
    uf.init(n);

    MSTEdge* h_mst_edges = (MSTEdge*)malloc((n - 1) * sizeof(MSTEdge));
    int mst_edge_count = 0;

    for (int i = 0; i < raw_edge_count && mst_edge_count < n - 1; i++) {
        if (uf.unite(h_raw_edges[i].u, h_raw_edges[i].v)) {
            h_mst_edges[mst_edge_count++] = h_raw_edges[i];
        }
    }

    // Flat clustering at threshold
    int num_clusters = n;
    for (int i = 0; i < mst_edge_count; i++) {
        if (h_mst_edges[i].dist <= cut_threshold) {
            num_clusters--;
        }
    }

    phase_end = std::chrono::high_resolution_clock::now();
    double t_cluster = std::chrono::duration<double>(phase_end - phase_start).count();

    // --- OUTPUT ---
    auto total_end = std::chrono::high_resolution_clock::now();
    double t_total = std::chrono::duration<double>(total_end - total_start).count();

    printf("\n=== RESULTS ===\n");
    printf("Loaded points             : %d\n", n);
    printf("Raw edges collected       : %d\n", raw_edge_count);
    printf("MST edges (after filter)  : %d / %d\n", mst_edge_count, n - 1);
    printf("Boruvka iterations        : %d\n", iteration);
    printf("Clusters (threshold %.1f) : %d\n", cut_threshold, num_clusters);
    printf("\nExecution time:\n");
    printf("   Grid construction  : %.4f s\n", t_grid);
    printf("   Device allocation  : %.4f s\n", t_alloc);
    printf("   Boruvka iterations : %.4f s\n", t_boruvka);
    printf("   Post-process       : %.4f s\n", t_cluster);
    printf("   TOTAL              : %.4f s\n", t_total);

    // Save linkage CSV
    std::ofstream out("linkage.csv");
    out << "idx1,idx2,dist\n";
    for (int i = 0; i < mst_edge_count; i++) {
        out << h_mst_edges[i].u << "," << h_mst_edges[i].v << ","
            << h_mst_edges[i].dist << "\n";
    }
    printf("\nSaved linkage.csv (%d edges)\n", mst_edge_count);

    // --- CLEANUP ---
    free(h_sorted_indices);
    free(h_cell_start);
    free(h_cell_count);
    free(h_comp_ids);
    free(h_raw_edges);
    free(h_mst_edges);
    uf.destroy();

    cudaFree(d_points);
    cudaFree(d_sorted_indices);
    cudaFree(d_cell_start);
    cudaFree(d_cell_count);
    cudaFree(d_comp_id);
    cudaFree(d_out_target);
    cudaFree(d_out_dist);
    cudaFree(d_comp_best);
    cudaFree(d_mst_edges);
    cudaFree(d_mst_count);
    cudaFree(d_changed);

    return 0;
}