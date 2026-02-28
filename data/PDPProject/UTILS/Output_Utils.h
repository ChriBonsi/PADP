#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <string>

/**
 * @brief Saves execution results to a CSV file for benchmark analysis.
 * * @param algo_name         The name of the algorithm (e.g., "seq_dbscan", "parallel_dbscan"). Used to name the file.
 * @param input_size        The number of points in the processed dataset.
 * @param num_clusters      The number of clusters identified.
 * @param threads_per_block Number of threads per block (set to 1 for sequential algorithms).
 * @param blocks_per_grid   Number of blocks per grid (set to 1 for sequential algorithms).
 * @param num_grids         Number of grids launched (typically 1).
 * @param execution_time_s  Total execution time in SECONDS.
 */
inline void logBenchmarkResult(
    const std::string& algo_name,
    size_t input_size,
    int num_clusters,
    int threads_per_block,
    int blocks_per_grid,
    int num_grids,
    double execution_time_s
) {
    // 1. Extract current date (YYYYMMDD format)
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d");
    std::string date_str = oss.str();
    
    // 2. Construct the output file path
    std::string out_filename = "/content/drive/MyDrive/PDPProject/OUTPUT/" + algo_name + ".csv";
    
    // 3. Check if the file exists to determine if the CSV header needs to be written
    std::ifstream check_file(out_filename);
    bool write_header = !check_file.is_open();
    check_file.close();
    
    // 4. Open the file in append mode and write the benchmark data
    std::ofstream out_file(out_filename, std::ios::app);
    if (out_file.is_open()) {
        if (write_header) {
            out_file << "Input_Size,Num_Clusters,Threads_Per_Block,Blocks_Per_Grid,Num_Grids,Execution_Time_s,Date\n";
        }
        
        out_file << input_size << ","
                 << num_clusters << ","
                 << threads_per_block << ","
                 << blocks_per_grid << ","
                 << num_grids << ","
                 << execution_time_s << ","
                 << date_str << "\n";
                 
        out_file.close();
        std::cout << "[IO] Benchmark saved to: " << out_filename << std::endl;
    } else {
        std::cerr << "[ERROR] Unable to create or write to file " << out_filename << std::endl;
    }
}

#endif // BENCHMARK_UTILS_H