#include <iostream>
#include <vector>
#include "utils/utils.h"
#include "conv/conv.h"
#include "batchnorm/batchnorm.h"
#include "relu/relu.h"
#include "maxpool/maxpool.h"
#include "fc/fc.h"

int main() {
    std::cout << "Final Stable Inference Pipeline\n";
    int N = 1;

    // INPUT
    auto x = read_binary("../data/input/input.bin");

    // ================= BLOCK 1 =================
    auto conv1 = conv2d(x,
        read_binary("../data/weights/conv1_weight.bin"),
        read_binary("../data/weights/conv1_bias.bin"),
        N, 3, 32, 32, 32, 3, 1, 1);

    auto bn1 = batchnorm2d(conv1,
        read_binary("../data/weights/bn1_weight.bin"),
        read_binary("../data/weights/bn1_bias.bin"),
        read_binary("../data/weights/bn1_running_mean.bin"),
        read_binary("../data/weights/bn1_running_var.bin"),
        N, 32, 32, 32);

    auto relu1 = relu(bn1);
    auto identity1 = relu1;

    // ================= BLOCK 2 =================
    auto conv2 = conv2d(relu1,
        read_binary("../data/weights/conv2_weight.bin"),
        read_binary("../data/weights/conv2_bias.bin"),
        N, 32, 32, 32, 64, 3, 1, 1);

    auto bn2 = batchnorm2d(conv2,
        read_binary("../data/weights/bn2_weight.bin"),
        read_binary("../data/weights/bn2_bias.bin"),
        read_binary("../data/weights/bn2_running_mean.bin"),
        read_binary("../data/weights/bn2_running_var.bin"),
        N, 64, 32, 32);

    auto relu2 = relu(bn2);
    auto pool1 = maxpool2d(relu2, N, 64, 32, 32, 2, 2);

    // ================= BLOCK 3 =================
    auto conv3 = conv2d(pool1,
        read_binary("../data/weights/conv3_weight.bin"),
        read_binary("../data/weights/conv3_bias.bin"),
        N, 64, 16, 16, 128, 3, 1, 1);

    auto bn3 = batchnorm2d(conv3,
        read_binary("../data/weights/bn3_weight.bin"),
        read_binary("../data/weights/bn3_bias.bin"),
        read_binary("../data/weights/bn3_running_mean.bin"),
        read_binary("../data/weights/bn3_running_var.bin"),
        N, 128, 16, 16);

    auto relu3 = relu(bn3);
    auto pool2 = maxpool2d(relu3, N, 128, 16, 16, 2, 2);

    // ================= SKIP 1 =================
    auto skip1 = conv2d(identity1,
        read_binary("../data/weights/skip1_0_weight.bin"),
        read_binary("../data/weights/skip1_0_bias.bin"),
        N, 32, 32, 32, 128, 1, 4, 0);

    skip1 = batchnorm2d(skip1,
        read_binary("../data/weights/skip1_1_weight.bin"),
        read_binary("../data/weights/skip1_1_bias.bin"),
        read_binary("../data/weights/skip1_1_running_mean.bin"),
        read_binary("../data/weights/skip1_1_running_var.bin"),
        N, 128, 8, 8);

    std::vector<float> block3(pool2.size());
    for (size_t i = 0; i < pool2.size(); i++) {
        block3[i] = pool2[i] + skip1[i];
    }

    block3 = relu(block3);

    // ================= BLOCK 4 =================
    auto conv4 = conv2d(block3,
        read_binary("../data/weights/conv4_weight.bin"),
        read_binary("../data/weights/conv4_bias.bin"),
        N, 128, 8, 8, 256, 3, 1, 1);

    auto bn4 = batchnorm2d(conv4,
        read_binary("../data/weights/bn4_weight.bin"),
        read_binary("../data/weights/bn4_bias.bin"),
        read_binary("../data/weights/bn4_running_mean.bin"),
        read_binary("../data/weights/bn4_running_var.bin"),
        N, 256, 8, 8);

    auto relu4 = relu(bn4);
    auto pool3 = maxpool2d(relu4, N, 256, 8, 8, 2, 2);

    // ================= SKIP 2 =================
    auto skip2 = conv2d(block3,
        read_binary("../data/weights/skip2_0_weight.bin"),
        read_binary("../data/weights/skip2_0_bias.bin"),
        N, 128, 8, 8, 256, 1, 2, 0);

    skip2 = batchnorm2d(skip2,
        read_binary("../data/weights/skip2_1_weight.bin"),
        read_binary("../data/weights/skip2_1_bias.bin"),
        read_binary("../data/weights/skip2_1_running_mean.bin"),
        read_binary("../data/weights/skip2_1_running_var.bin"),
        N, 256, 4, 4);

    std::vector<float> block4(pool3.size());
    for (size_t i = 0; i < pool3.size(); i++) {
        block4[i] = pool3[i] + skip2[i];
    }

    block4 = relu(block4);

    // Optional debug
    write_binary("../data/reference/block4_cpp.bin", block4);

    // ================= FLATTEN =================
    std::vector<float> flat(256 * 4 * 4);

    int idx = 0;

    // NCHW flatten (PyTorch exact)
    for (int c = 0; c < 256; c++) {
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {

                int index = c * 4 * 4 + h * 4 + w;

                flat[idx++] = block4[index];
            }
        }
    }

    // FC1
    auto fc1 = linear(flat,
        read_binary("../data/weights/fc1_weight.bin"),
        read_binary("../data/weights/fc1_bias.bin"),
        4096, 256);

    // ReLU after FC1
    auto relu_fc1 = relu(fc1);

    // SAVE THIS (size 256)
    write_binary("../data/reference/relu_final_cpp.bin", relu_fc1);

    // FC2
    auto fc2 = linear(relu_fc1,
        read_binary("../data/weights/fc2_weight.bin"),
        read_binary("../data/weights/fc2_bias.bin"),
        256, 10);

    // Final output
    write_binary("../data/reference/final_output_cpp.bin", fc2);

    std::cout << "Pipeline SUCCESS\n";

    return 0;

}
