#include <iostream>
#include "inference.h"
#include "utils/utils.h"
#include "conv/conv.h"
#include "batchnorm/batchnorm.h"
#include "relu/relu.h"
#include "maxpool/maxpool.h"
#include "fc/fc.h"

// ================= LAZY LOAD =================
static bool initialized = false;

// Block 1
static std::vector<float> conv1_w, conv1_b;
static std::vector<float> bn1_w, bn1_b, bn1_m, bn1_v;

// Block 2
static std::vector<float> conv2_w, conv2_b;
static std::vector<float> bn2_w, bn2_b, bn2_m, bn2_v;

// Block 3
static std::vector<float> conv3_w, conv3_b;
static std::vector<float> bn3_w, bn3_b, bn3_m, bn3_v;

// Block 4
static std::vector<float> conv4_w, conv4_b;
static std::vector<float> bn4_w, bn4_b, bn4_m, bn4_v;

// Skip connections
static std::vector<float> skip1_w, skip1_b, skip1_bn_w, skip1_bn_b, skip1_bn_m, skip1_bn_v;
static std::vector<float> skip2_w, skip2_b, skip2_bn_w, skip2_bn_b, skip2_bn_m, skip2_bn_v;

// FC
static std::vector<float> fc1_w, fc1_b;
static std::vector<float> fc2_w, fc2_b;

// ================= LOAD FUNCTION =================
void load_weights_once() {
    if (initialized) return;

    std::cout << "Loading weights...\n";

    conv1_w = read_binary("data/weights/conv1_weight.bin");
    conv1_b = read_binary("data/weights/conv1_bias.bin");

    bn1_w = read_binary("data/weights/bn1_weight.bin");
    bn1_b = read_binary("data/weights/bn1_bias.bin");
    bn1_m = read_binary("data/weights/bn1_running_mean.bin");
    bn1_v = read_binary("data/weights/bn1_running_var.bin");

    conv2_w = read_binary("data/weights/conv2_weight.bin");
    conv2_b = read_binary("data/weights/conv2_bias.bin");

    bn2_w = read_binary("data/weights/bn2_weight.bin");
    bn2_b = read_binary("data/weights/bn2_bias.bin");
    bn2_m = read_binary("data/weights/bn2_running_mean.bin");
    bn2_v = read_binary("data/weights/bn2_running_var.bin");

    conv3_w = read_binary("data/weights/conv3_weight.bin");
    conv3_b = read_binary("data/weights/conv3_bias.bin");

    bn3_w = read_binary("data/weights/bn3_weight.bin");
    bn3_b = read_binary("data/weights/bn3_bias.bin");
    bn3_m = read_binary("data/weights/bn3_running_mean.bin");
    bn3_v = read_binary("data/weights/bn3_running_var.bin");

    conv4_w = read_binary("data/weights/conv4_weight.bin");
    conv4_b = read_binary("data/weights/conv4_bias.bin");

    bn4_w = read_binary("data/weights/bn4_weight.bin");
    bn4_b = read_binary("data/weights/bn4_bias.bin");
    bn4_m = read_binary("data/weights/bn4_running_mean.bin");
    bn4_v = read_binary("data/weights/bn4_running_var.bin");

    // Skip1
    skip1_w = read_binary("data/weights/skip1_0_weight.bin");
    skip1_b = read_binary("data/weights/skip1_0_bias.bin");
    skip1_bn_w = read_binary("data/weights/skip1_1_weight.bin");
    skip1_bn_b = read_binary("data/weights/skip1_1_bias.bin");
    skip1_bn_m = read_binary("data/weights/skip1_1_running_mean.bin");
    skip1_bn_v = read_binary("data/weights/skip1_1_running_var.bin");

    // Skip2
    skip2_w = read_binary("data/weights/skip2_0_weight.bin");
    skip2_b = read_binary("data/weights/skip2_0_bias.bin");
    skip2_bn_w = read_binary("data/weights/skip2_1_weight.bin");
    skip2_bn_b = read_binary("data/weights/skip2_1_bias.bin");
    skip2_bn_m = read_binary("data/weights/skip2_1_running_mean.bin");
    skip2_bn_v = read_binary("data/weights/skip2_1_running_var.bin");

    fc1_w = read_binary("data/weights/fc1_weight.bin");
    fc1_b = read_binary("data/weights/fc1_bias.bin");

    fc2_w = read_binary("data/weights/fc2_weight.bin");
    fc2_b = read_binary("data/weights/fc2_bias.bin");

    initialized = true;
    std::cout << "Weights loaded successfully\n";
}

// ================= MAIN =================
std::vector<float> run_inference(const std::vector<float>& x) {

    load_weights_once();

    int N = 1;

    // BLOCK 1
    auto conv1 = conv2d(x, conv1_w, conv1_b, N, 3, 32, 32, 32, 3, 1, 1);
    auto bn1 = batchnorm2d(conv1, bn1_w, bn1_b, bn1_m, bn1_v, N, 32, 32, 32);
    auto relu1 = relu(bn1);

    // BLOCK 2
    auto conv2 = conv2d(relu1, conv2_w, conv2_b, N, 32, 32, 32, 64, 3, 1, 1);
    auto bn2 = batchnorm2d(conv2, bn2_w, bn2_b, bn2_m, bn2_v, N, 64, 32, 32);
    auto relu2 = relu(bn2);
    auto pool1 = maxpool2d(relu2, N, 64, 32, 32, 2, 2);

    // BLOCK 3
    auto conv3 = conv2d(pool1, conv3_w, conv3_b, N, 64, 16, 16, 128, 3, 1, 1);
    auto bn3 = batchnorm2d(conv3, bn3_w, bn3_b, bn3_m, bn3_v, N, 128, 16, 16);
    auto relu3 = relu(bn3);
    auto pool2 = maxpool2d(relu3, N, 128, 16, 16, 2, 2);

    // SKIP 1
    auto skip1 = conv2d(relu1, skip1_w, skip1_b, N, 32, 32, 32, 128, 1, 4, 0);
    skip1 = batchnorm2d(skip1, skip1_bn_w, skip1_bn_b, skip1_bn_m, skip1_bn_v, N, 128, 8, 8);

    for (size_t i = 0; i < pool2.size(); i++) {
        pool2[i] += skip1[i];
    }
    pool2 = relu(pool2);

    // BLOCK 4
    auto conv4 = conv2d(pool2, conv4_w, conv4_b, N, 128, 8, 8, 256, 3, 1, 1);
    auto bn4 = batchnorm2d(conv4, bn4_w, bn4_b, bn4_m, bn4_v, N, 256, 8, 8);
    auto relu4 = relu(bn4);
    auto pool3 = maxpool2d(relu4, N, 256, 8, 8, 2, 2);

    // SKIP 2
    auto skip2 = conv2d(pool2, skip2_w, skip2_b, N, 128, 8, 8, 256, 1, 2, 0);
    skip2 = batchnorm2d(skip2, skip2_bn_w, skip2_bn_b, skip2_bn_m, skip2_bn_v, N, 256, 4, 4);

    for (size_t i = 0; i < pool3.size(); i++) {
        pool3[i] += skip2[i];
    }
    pool3 = relu(pool3);

    // FC
    auto flat = pool3;

    auto fc1 = linear(flat, fc1_w, fc1_b, 4096, 256);
    fc1 = relu(fc1);

    auto fc2 = linear(fc1, fc2_w, fc2_b, 256, 10);

    return fc2;
}