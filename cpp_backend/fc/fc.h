#pragma once
#include <vector>

std::vector<float> linear(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int in_features,
    int out_features
);