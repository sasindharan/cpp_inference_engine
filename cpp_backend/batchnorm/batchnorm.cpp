#include "batchnorm.h"
#include <cmath>

inline int idx4D(int n, int c, int h, int w,
                 int C, int H, int W) {
    return n * C * H * W + c * H * W + h * W + w;
}

std::vector<float> batchnorm2d(
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    const std::vector<float>& mean,
    const std::vector<float>& var,
    int N, int C, int H, int W,
    float eps
) {
    std::vector<float> output(input.size());

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {

            float g = gamma[c];
            float b = beta[c];
            float m = mean[c];
            float v = var[c];

            float denom = 1.0f / std::sqrt(v + eps);

            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {

                    int index = idx4D(n, c, h, w, C, H, W);

                    float x = input[index];

                    output[index] = g * (x - m) * denom + b;
                }
            }
        }
    }

    return output;
}