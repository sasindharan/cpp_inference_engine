#pragma once
#include <vector>
#include <string>

// File I/O
std::vector<float> read_binary(const std::string& path);
void write_binary(const std::string& path, const std::vector<float>& data);

// Comparison
void compare_outputs(const std::vector<float>& a,
                     const std::vector<float>& b,
                     float atol = 1e-5);

// Logging
void log_info(const std::string& msg);
void log_error(const std::string& msg);