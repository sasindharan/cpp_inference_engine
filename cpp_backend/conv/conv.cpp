#include "conv.h"

// Convert 4D index → 1D
inline int idx4D(int n, int c, int h, int w,
                 int C, int H, int W) {
    return n * C * H * W + c * H * W + h * W + w;
}

// Weight index
inline int widx(int oc, int ic, int kh, int kw,
                int in_channels, int kernel) {
    return oc * in_channels * kernel * kernel +
           ic * kernel * kernel +
           kh * kernel + kw;
}

std::vector<float> conv2d(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    int N, int C, int H, int W,
    int out_channels,
    int kernel,
    int stride,
    int padding
) {
    int H_out = (H + 2 * padding - kernel) / stride + 1;
    int W_out = (W + 2 * padding - kernel) / stride + 1;

    std::vector<float> output(N * out_channels * H_out * W_out, 0.0f);

    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {

                    float sum = bias[oc];

                    for (int ic = 0; ic < C; ic++) {
                        for (int kh = 0; kh < kernel; kh++) {
                            for (int kw = 0; kw < kernel; kw++) {

                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                // boundary check (padding)
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {

                                    int input_idx = idx4D(n, ic, ih, iw, C, H, W);
                                    int weight_idx = widx(oc, ic, kh, kw, C, kernel);

                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }

                    int out_idx = idx4D(n, oc, oh, ow,
                                        out_channels, H_out, W_out);

                    output[out_idx] = sum;
                }
            }
        }
    }

    return output;
}