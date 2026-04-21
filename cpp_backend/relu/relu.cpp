#include "relu.h"

std::vector<float> relu(const std::vector<float>& input) {

    std::vector<float> output(input.size());

    for (size_t i = 0; i < input.size(); i++) {

        // ReLU: max(0, x)
        if (input[i] > 0.0f)
            output[i] = input[i];
        else
            output[i] = 0.0f;
    }

    return output;
}