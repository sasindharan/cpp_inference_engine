#include "softmax.h"
#include <cmath>
#include <algorithm>

std::vector<float> softmax(const std::vector<float>& input) {

    std::vector<float> output(input.size());

    float max_val = *std::max_element(input.begin(), input.end());

    float sum = 0.0f;

    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (size_t i = 0; i < input.size(); i++) {
        output[i] /= sum;
    }

    return output;
}