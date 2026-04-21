#pragma once
#include <vector>

std::vector<float> conv2d(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    int N, int C, int H, int W,
    int out_channels,
    int kernel_size,
    int stride,
    int padding
);