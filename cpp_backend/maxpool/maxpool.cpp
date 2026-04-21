#include "maxpool.h"
#include <algorithm> // for std::max

inline int idx4D(int n, int c, int h, int w,
                 int C, int H, int W) {
    return n * C * H * W + c * H * W + h * W + w;
}

std::vector<float> maxpool2d(
    const std::vector<float>& input,
    int N, int C, int H, int W,
    int kernel,
    int stride
) {
    int H_out = (H - kernel) / stride + 1;
    int W_out = (W - kernel) / stride + 1;

    std::vector<float> output(N * C * H_out * W_out, 0.0f);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {

                    float max_val = -1e9f; // very small number

                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {

                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;

                            int index = idx4D(n, c, ih, iw, C, H, W);

                            max_val = std::max(max_val, input[index]);
                        }
                    }

                    int out_idx = idx4D(n, c, oh, ow, C, H_out, W_out);
                    output[out_idx] = max_val;
                }
            }
        }
    }

    return output;
}