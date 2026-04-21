#include "fc.h"
#include <vector>

std::vector<float> linear(
const std::vector<float>& input,
const std::vector<float>& weight,
const std::vector<float>& bias,
int in_features,
int out_features
) {
std::vector<float> output(out_features, 0.0f);

for (int o = 0; o < out_features; o++) {
    float sum = 0.0f;

    for (int i = 0; i < in_features; i++) {
        sum += input[i] * weight[o * in_features + i];
    }

    output[o] = sum + bias[o];
}

return output;

}
