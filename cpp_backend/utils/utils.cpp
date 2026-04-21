#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

// -------- READ --------
std::vector<float> read_binary(const std::string& path) {

    std::ifstream file(path, std::ios::binary);

    if (!file) {
        std::cout << "[ERROR] Cannot open: " << path << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(float);
    file.seekg(0, std::ios::beg);

    std::vector<float> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));

    std::cout << "[LOAD] " << path << " size=" << size << std::endl;

    return data;
}

// -------- WRITE --------
void write_binary(const std::string& path, const std::vector<float>& data) {

    std::ofstream file(path, std::ios::binary);

    if (!file) {
        std::cout << "[ERROR] Cannot write: " << path << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(data.data()),
               data.size() * sizeof(float));

    std::cout << "[SAVE] " << path << " size=" << data.size() << std::endl;
}

// -------- COMPARE --------
void compare_outputs(const std::vector<float>& a,
                     const std::vector<float>& b,
                     float atol) {

    if (a.size() != b.size()) {
        std::cout << "[ERROR] Size mismatch!\n";
        return;
    }

    float max_diff = 0.0f;
    float mean_diff = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
    }

    mean_diff /= a.size();

    std::cout << "\n===== OUTPUT CHECK =====\n";
    std::cout << "Max diff  : " << max_diff << std::endl;
    std::cout << "Mean diff : " << mean_diff << std::endl;

    if (max_diff < atol)
        std::cout << "OUTPUT MATCH\n";
    else
        std::cout << "OUTPUT MISMATCH \n";
}

// -------- LOG --------
void log_info(const std::string& msg) {
    std::cout << "[INFO] " << msg << std::endl;
}

void log_error(const std::string& msg) {
    std::cout << "[ERROR] " << msg << std::endl;
}