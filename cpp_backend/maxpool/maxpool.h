#pragma once
#include <vector>

std::vector<float> maxpool2d(
    const std::vector<float>& input,
    int N, int C, int H, int W,
    int kernel,
    int stride
);