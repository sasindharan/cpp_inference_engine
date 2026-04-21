#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>

#include "json.hpp"

#include "utils/utils.h"
#include "conv/conv.h"
#include "batchnorm/batchnorm.h"
#include "relu/relu.h"
#include "maxpool/maxpool.h"
#include "fc/fc.h"
#include "softmax/softmax.h"

using json = nlohmann::json;

std::map<std::string, std::vector<float>> tensors;

// ---------- LOAD ----------
std::vector<float> get_tensor(const std::string& path) {

    if (tensors.count(path)) return tensors[path];

    auto data = read_binary(path);

    if (data.empty()) {
        log_error("Failed to load: " + path);
        exit(-1);
    }

    tensors[path] = data;
    return data;
}

// ---------- SAVE ----------
void set_tensor(const std::string& path, const std::vector<float>& data) {
    tensors[path] = data;
    write_binary(path, data);
}

// ---------- MODEL EXEC ----------
void run_model(const json& config) {

    tensors.clear();

    for (auto& layer : config["layers"]) {

        std::string name = layer["name"];
        std::string type = layer["type"];

        log_info("Running: " + name);

        // -------- CONV --------
        if (type == "conv") {

            auto input = get_tensor(layer["input"]);
            auto w = read_binary(layer["weights"]);
            auto b = read_binary(layer["bias"]);

            auto s = layer["input_shape"];
            int N=s[0],C=s[1],H=s[2],W=s[3];

            int OC = layer["output_shape"][1];
            int K = layer["kernel_size"];
            int S = layer["stride"];
            int P = layer["padding"];

            auto out = conv2d(input, w, b, N,C,H,W, OC,K,S,P);
            set_tensor(layer["output"], out);
        }

        // -------- BN --------
        else if (type == "batchnorm") {

            auto input = get_tensor(layer["input"]);

            auto w = read_binary(layer["weight"]);
            auto b = read_binary(layer["bias"]);
            auto m = read_binary(layer["running_mean"]);
            auto v = read_binary(layer["running_var"]);

            auto s = layer["input_shape"];
            int N=s[0],C=s[1],H=s[2],W=s[3];

            auto out = batchnorm2d(input,w,b,m,v,N,C,H,W);
            set_tensor(layer["output"], out);
        }

        // -------- RELU --------
        else if (type == "relu") {

            auto out = relu(get_tensor(layer["input"]));
            set_tensor(layer["output"], out);
        }

        // -------- MAXPOOL --------
        else if (type == "maxpool") {

            auto input = get_tensor(layer["input"]);

            auto s = layer["input_shape"];
            int N=s[0],C=s[1],H=s[2],W=s[3];

            int K = layer["kernel_size"];
            int S = layer["stride"];

            auto out = maxpool2d(input,N,C,H,W,K,S);
            set_tensor(layer["output"], out);
        }

        // -------- ADD --------
        else if (type == "add") {

            auto a = get_tensor(layer["input1"]);
            auto b = get_tensor(layer["input2"]);

            std::vector<float> out(a.size());

            for (size_t i=0;i<a.size();i++)
                out[i] = a[i] + b[i];

            set_tensor(layer["output"], out);
        }

        // -------- FC --------
        else if (type == "fc") {

            auto input = get_tensor(layer["input"]);

            auto w = read_binary(layer["weights"]);
            auto b = read_binary(layer["bias"]);

            int in_f = layer["input_shape"][1];
            int out_f = layer["output_shape"][1];

            auto out = linear(input,w,b,in_f,out_f);
            set_tensor(layer["output"], out);
        }

        // -------- SOFTMAX --------
        else if (type == "softmax") {

            auto input = get_tensor(layer["input"]);
            auto out = softmax(input);

            set_tensor(layer["output"], out);
        }

        else {
            log_error("Unknown layer: " + type);
            exit(-1);
        }
    }

    log_info("MODEL COMPLETE");
}

// ---------- MAIN ----------
int main(int argc, char** argv) {

    std::string config_path = "configs/model.json";

    // test mode
    if (argc > 1 && std::string(argv[1]) == "test") {
        config_path = "configs/softmax_test.json";
    }

    std::ifstream f(config_path);

    if (!f.is_open()) {
        log_error("Cannot open config: " + config_path);
        return -1;
    }

    json config;
    f >> config;

    run_model(config);

    return 0;
}