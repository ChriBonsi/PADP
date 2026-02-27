#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <string>

/**
 * @brief Salva i risultati di esecuzione in un file CSV per l'analisi dei benchmark.
 * * @param algo_name Il nome dell'algoritmo (es. "seq_dbscan", "parallel_dbscan"). Definisce il nome del file.
 * @param input_size Il numero di punti nel dataset processato.
 * @param num_clusters Il numero di cluster identificati.
 * @param threads_per_block Numero di threads per blocco (1 per algoritmi sequenziali).
 * @param blocks_per_grid Numero di blocchi per grid (1 per algoritmi sequenziali).
 * @param num_grids Numero di grid lanciate (tipicamente 1).
 * @param execution_time_s Tempo totale di esecuzione in SECONDI.
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
    // 1. Estrazione data corrente (YYYYMMDD)
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d");
    std::string date_str = oss.str();
    
    // 2. Costruzione path file
    std::string out_filename = "/content/drive/MyDrive/PDPProject/OUTPUT/" + algo_name + "_" + date_str + ".csv";
    
    // 3. Controllo file per l'intestazione
    std::ifstream check_file(out_filename);
    bool write_header = !check_file.is_open();
    check_file.close();
    
    // 4. Scrittura su file in append
    std::ofstream out_file(out_filename, std::ios::app);
    if (out_file.is_open()) {
        if (write_header) {
            out_file << "Input_Size,Num_Clusters,Threads_Per_Block,Blocks_Per_Grid,Num_Grids,Execution_Time_s\n";
        }
        
        out_file << input_size << ","
                 << num_clusters << ","
                 << threads_per_block << ","
                 << blocks_per_grid << ","
                 << num_grids << ","
                 << execution_time_s << "\n";
                 
        out_file.close();
        std::cout << "[IO] Benchmark saved to: " << out_filename << std::endl;
    } else {
        std::cerr << "[ERROR] Unable to create or write to file " << out_filename << std::endl;
    }
}

#endif // BENCHMARK_UTILS_H