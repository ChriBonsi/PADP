#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <sstream>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// Error checking macro
#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define BLOCK_SIZE 256

// --- DATA STRUCTURES ---

// Simple 3D point structure (Equivalent to 'Star' in your snippet)
struct Point3D {
    float x, y, z;
};

// Graph structure to represent the adjacency list in CSR format
struct Graph {
    thrust::device_vector<int> row_ptr;      // Row pointer for CSR format
    thrust::device_vector<int> col_indices;  // Column indices for CSR format
};

// --- DATA LOADING FUNCTION ---

// Function to load the CSV dataset
// Allocates host memory (malloc) and device memory (cudaMalloc), then copies host -> device
// Returns the number of points loaded
int load_dataset(const std::string& filename, Point3D** h_points_ptr, Point3D** d_points_ptr) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Impossible to open file " << filename << std::endl;
        return 0;
    }

    // Read into a standard host vector first for parsing simplicity
    std::vector<Point3D> temp_points;
    std::string line;

    // Skip the header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        Point3D p;

        // CSV format: source_id, x, y, z, phot...
        // Skip source_id
        std::getline(ss, cell, ',');

        // Read X
        std::getline(ss, cell, ',');
        try { p.x = std::stof(cell); } catch (...) { p.x = 0.0f; }

        // Read Y
        std::getline(ss, cell, ',');
        try { p.y = std::stof(cell); } catch (...) { p.y = 0.0f; }

        // Read Z
        std::getline(ss, cell, ',');
        try { p.z = std::stof(cell); } catch (...) { p.z = 0.0f; }

        temp_points.push_back(p);
    }

    int num_points = temp_points.size();
    printf("Read %d points from CSV.\n", num_points);

    // Host memory allocation
    *h_points_ptr = (Point3D*)malloc(num_points * sizeof(Point3D));
    memcpy(*h_points_ptr, temp_points.data(), num_points * sizeof(Point3D));

    // Device memory allocation and copy host -> device
    CHECK_CUDA(cudaMalloc(d_points_ptr, num_points * sizeof(Point3D)));
    CHECK_CUDA(cudaMemcpy(*d_points_ptr, *h_points_ptr, num_points * sizeof(Point3D), cudaMemcpyHostToDevice));

    return num_points;
}


// --- KERNELS ---

// Device function to compute Euclidean distance between two 3D points
__device__ float euclidean_distance(const Point3D& a, const Point3D& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// Auxiliary kernel: Count neighbors for each point to prepare row_ptr (CSR)
// Without this step, construct_adjacency_list doesn't know where to write.
__global__ void count_neighbors_kernel(const Point3D* points, int* degrees, int num_points, float eps) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_points) {
        int count = 0;
        for (int j = 0; j < num_points; ++j) {
            if (euclidean_distance(points[idx], points[j]) <= eps) {
                count++;
            }
        }
        degrees[idx] = count;
    }
}

// KERNEL 1: Graph Construction (Adjacency List in CSR format)
// Replaces the previous adjacency matrix approach.
// Fills col_indices using the offsets computed in row_ptr.
__global__ void construct_adjacency_list(const Point3D* points, int* row_ptr, int* col_indices,
                                         int num_points, float eps) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_points) {
        int count = 0;
        for (int j = 0; j < num_points; ++j) {
            if (euclidean_distance(points[idx], points[j]) <= eps) { ////////////////////////////////////////////////
                col_indices[row_ptr[idx] + count++] = j;
            }
        }
    }
}

// KERNEL 2: Identify Core Points (Degree Calculation)
// Uses CSR row_ptr to count neighbors: degree[i] = row_ptr[i+1] - row_ptr[i]
__global__ void identify_core_points_kernel(const int* row_ptr, int* node_type, int num_points, int min_pts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) return;

    int neighbors = row_ptr[idx + 1] - row_ptr[idx];

    // Node Type: 0 = Noise/Border (initially), 1 = Core
    if (neighbors >= min_pts) {
        node_type[idx] = 1; // Core
    } else {
        node_type[idx] = 0; // Non-Core
    }
}

// KERNEL 3: BFS Propagation (Iterative Step)
// Based on G-DBSCAN Paper Section 3.2 (Graph Traversal)
// Propagates the cluster ID from the 'current_frontier' to connected neighbors.
// Adapted to use CSR adjacency list instead of adjacency matrix.
__global__ void bfs_propagate_kernel(const int* row_ptr,
                                     const int* col_indices,
                                     int* cluster_ids,
                                     bool* current_frontier,
                                     bool* next_frontier,
                                     const int* node_type,
                                     int num_points,
                                     int current_cluster_id,
                                     bool* d_continue) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    // If I am in the frontier, I must propagate my cluster to my neighbors
    if (current_frontier[tid]) {
        // Iterate over actual neighbors from CSR adjacency list
        for (int e = row_ptr[tid]; e < row_ptr[tid + 1]; ++e) {
            int neighbor = col_indices[e];

            // If not yet visited (cluster_id == 0)
            if (cluster_ids[neighbor] == 0) {

                // Assign Cluster ID
                cluster_ids[neighbor] = current_cluster_id;

                // If the neighbor is a CORE point, it continues the expansion (add to next frontier)
                // If it's a border point, it gets the ID but stops expansion.
                if (node_type[neighbor] == 1) {
                    next_frontier[neighbor] = true;
                    *d_continue = true; // Signal host to continue loop
                }
            }
        }
        // Remove myself from current frontier
        current_frontier[tid] = false;
    }
}


// --- HOST FUNCTIONS ---

int main() {
    // 1. Setup Parameters
    // std::string GAIA_CARTESIAN_FILENAME = "GAIA_DR3_Cartesian_Heliocentric.csv";
    std::string GAIA_CARTESIAN_FILENAME = "GAIA_nearest_10000.csv";

    std::string filename = "/content/drive/MyDrive/PDPProject/" + GAIA_CARTESIAN_FILENAME;
    // IMPORTANT: Ensure this file is present in the current directory!

    float epsilon = 2.0f;
    int min_pts = 7;

    printf("G-DBSCAN Simulation (Graph-Based, Adjacency List / CSR)\n");
    printf("Dataset: %s\n", filename.c_str());
    printf("Parameters: Epsilon = %.2f, MinPts = %d\n", epsilon, min_pts);

    // 2. Data Loading
    Point3D* h_points = nullptr;
    Point3D* d_points = nullptr;

    // Use the custom function to load data
    int num_points = load_dataset(filename, &h_points, &d_points);

    if (num_points == 0) {
        printf("Failed to load dataset. Exiting.\n");
        return 1;
    }

    // 3. Memory Allocation (Auxiliary structures)
    int* d_node_type;
    int* d_cluster_ids;
    bool* d_frontier_a; // Ping-pong buffers for BFS
    bool* d_frontier_b;
    bool* d_continue;

    CHECK_CUDA(cudaMalloc(&d_node_type, num_points * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cluster_ids, num_points * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_frontier_a, num_points * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_frontier_b, num_points * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_continue, sizeof(bool)));

    // Initialize output arrays
    CHECK_CUDA(cudaMemset(d_cluster_ids, 0, num_points * sizeof(int))); // 0 = Unvisited/Noise
    CHECK_CUDA(cudaMemset(d_frontier_a, 0, num_points * sizeof(bool)));
    CHECK_CUDA(cudaMemset(d_frontier_b, 0, num_points * sizeof(bool)));

    // --- TIMING SETUP ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- PHASE 1: GRAPH CONSTRUCTION (Adjacency List / CSR) ---
    int block1D = BLOCK_SIZE;
    int grid1D = (num_points + block1D - 1) / block1D;

    // Step 1a: Count neighbors (degrees) for each point
    int* d_degrees;
    CHECK_CUDA(cudaMalloc(&d_degrees, num_points * sizeof(int)));

    printf("Phase 1a: Counting neighbors (degrees)...\n");
    count_neighbors_kernel<<<grid1D, block1D>>>(d_points, d_degrees, num_points, epsilon);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy degrees to host (needed for computing total edges)
    int* h_degrees = (int*)malloc(num_points * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_degrees, d_degrees, num_points * sizeof(int), cudaMemcpyDeviceToHost));

    // Step 1b: Prefix sum (exclusive scan) on degrees to build row_ptr
    Graph g;
    g.row_ptr.resize(num_points + 1);

    thrust::device_ptr<int> dev_degrees(d_degrees);
    thrust::exclusive_scan(dev_degrees, dev_degrees + num_points, g.row_ptr.begin());

    // Compute total number of edges and set the last element of row_ptr
    int total_edges = g.row_ptr[num_points - 1] + h_degrees[num_points - 1];
    g.row_ptr[num_points] = total_edges;

    printf("Phase 1b: Prefix sum complete. Total edges: %d\n", total_edges);

    // Step 1c: Build the adjacency list (col_indices)
    g.col_indices.resize(total_edges);

    printf("Phase 1c: Building Adjacency List...\n");
    construct_adjacency_list<<<grid1D, block1D>>>(
        d_points,
        thrust::raw_pointer_cast(g.row_ptr.data()),
        thrust::raw_pointer_cast(g.col_indices.data()),
        num_points,
        epsilon
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Get raw pointers for subsequent kernels
    int* d_row_ptr = thrust::raw_pointer_cast(g.row_ptr.data());
    int* d_col_indices = thrust::raw_pointer_cast(g.col_indices.data());

    // --- PHASE 2: CORE POINT IDENTIFICATION ---
    printf("Phase 2: Identifying Core Points...\n");
    identify_core_points_kernel<<<grid1D, block1D>>>(d_row_ptr, d_node_type, num_points, min_pts);
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- PHASE 3: CLUSTERING (BFS TRAVERSAL) ---

    //Gestione della Frontiera BFS inefficiente
    //Nel loop do-while della BFS:
    //CHECK_CUDA(cudaMemcpy(d_frontier_a, d_frontier_b, num_points * sizeof(bool), cudaMemcpyDeviceToDevice));
    //CHECK_CUDA(cudaMemset(d_frontier_b, 0, num_points * sizeof(bool)));
    //Stai copiando e azzerando interi array (che possono essere grandi) ad ogni passo dell'espansione.
    //Soluzione (Double Buffering): Invece di copiare i dati, scambia semplicemente i puntatori d_frontier_a e d_frontier_b ad ogni iterazione. È un'operazione istantanea.

    printf("Phase 3: Graph Traversal (BFS)...\n");

    // Copiamo node_type su Host una volta sola
    std::vector<int> h_node_type(num_points);
    CHECK_CUDA(cudaMemcpy(h_node_type.data(), d_node_type, num_points * sizeof(int), cudaMemcpyDeviceToHost));

    // Manteniamo una copia locale dei cluster_ids per evitare di interrogare la GPU punto per punto
    std::vector<int> h_cluster_ids(num_points, 0);

    int current_cluster_id = 1;
    bool h_continue = false;

    // Puntatori temporanei per lo swapping
    bool* d_front_in = d_frontier_a;
    bool* d_front_out = d_frontier_b;

    for (int i = 0; i < num_points; ++i) {
        // Controllo VELOCE su Host usando la copia locale
        if (h_node_type[i] == 1 && h_cluster_ids[i] == 0) {

            // Trovato un nuovo seed (Core point non visitato)
            // 1. Aggiorniamo Host
            h_cluster_ids[i] = current_cluster_id;

            // 2. Aggiorniamo Device (Seed e Frontiera iniziale)
            CHECK_CUDA(cudaMemcpy(&d_cluster_ids[i], &current_cluster_id, sizeof(int), cudaMemcpyHostToDevice));

            bool true_val = true;
            CHECK_CUDA(cudaMemcpy(&d_front_in[i], &true_val, sizeof(bool), cudaMemcpyHostToDevice)); // Attiva seed

            // 3. BFS Loop
            do {
                h_continue = false;
                CHECK_CUDA(cudaMemcpy(d_continue, &h_continue, sizeof(bool), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemset(d_front_out, 0, num_points * sizeof(bool))); // Pulisci output

                bfs_propagate_kernel<<<grid1D, block1D>>>(
                    d_row_ptr,
                    d_col_indices,
                    d_cluster_ids,
                    d_front_in,  // Leggi da qui
                    d_front_out, // Scrivi qui
                    d_node_type,
                    num_points,
                    current_cluster_id,
                    d_continue
                );
                CHECK_CUDA(cudaDeviceSynchronize()); // Importante per la logica di swap

                // SWAP DEI PUNTATORI (Double Buffering) - Zero costo di copia
                std::swap(d_front_in, d_front_out);

                CHECK_CUDA(cudaMemcpy(&h_continue, d_continue, sizeof(bool), cudaMemcpyDeviceToHost));

            } while (h_continue);

            // 4. Sincronizziamo la mappa dei cluster su Host ALLA FINE del cluster
            // Questo costa una sola copia grande invece di migliaia piccole.
            CHECK_CUDA(cudaMemcpy(h_cluster_ids.data(), d_cluster_ids, num_points * sizeof(int), cudaMemcpyDeviceToHost));

            current_cluster_id++;

            // Pulisci i buffer di frontiera per il prossimo cluster (sicurezza)
            CHECK_CUDA(cudaMemset(d_frontier_a, 0, num_points * sizeof(bool)));
            CHECK_CUDA(cudaMemset(d_frontier_b, 0, num_points * sizeof(bool)));
            d_front_in = d_frontier_a; // Reset puntatori base
            d_front_out = d_frontier_b;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Execution Time: %.3f ms\n", ms);
    printf("Total Clusters Found: %d\n", current_cluster_id - 1);

    // --- CLEANUP ---
    if(h_points) free(h_points);
    if(h_degrees) free(h_degrees);
    if(d_points) cudaFree(d_points);
    cudaFree(d_degrees);
    cudaFree(d_node_type);
    cudaFree(d_cluster_ids);
    cudaFree(d_frontier_a);
    cudaFree(d_frontier_b);
    cudaFree(d_continue);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}