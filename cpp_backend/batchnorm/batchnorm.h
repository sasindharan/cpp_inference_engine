#pragma once
#include <vector>

std::vector<float> batchnorm2d(
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    const std::vector<float>& mean,
    const std::vector<float>& var,
    int N, int C, int H, int W,
    float eps = 1e-5f
);