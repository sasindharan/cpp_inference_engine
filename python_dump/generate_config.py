import torch
import json
from model import ResNet

CONFIG_PATH = "../configs/model.json"

def main():
    model = ResNet()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    layers = []
    prev_output = "data/input/input.bin"

    # Track special tensors
    identity1 = None
    identity2 = None

    for name, module in model.named_modules():

        clean = name.replace(".", "_")

        # ================= CONV =================
        if isinstance(module, torch.nn.Conv2d):

            out = f"data/output/{clean}_output.bin"

            layer = {
                "name": clean,
                "type": "conv",
                "input": prev_output,
                "output": out,
                "weights": f"data/weights/{clean}_weight.bin",
                "bias": f"data/weights/{clean}_bias.bin",
                "kernel_size": module.kernel_size[0],
                "stride": module.stride[0],
                "padding": module.padding[0],
                "in_channels": module.in_channels,
                "out_channels": module.out_channels
            }

            layers.append(layer)
            prev_output = out

            # SAVE identity1 after conv1
            if clean == "conv1":
                identity1 = prev_output

        # ================= RELU =================
        elif isinstance(module, torch.nn.ReLU):

            out = f"data/output/{clean}_output.bin"

            layers.append({
                "name": clean,
                "type": "relu",
                "input": prev_output,
                "output": out
            })

            prev_output = out

        # ================= MAXPOOL =================
        elif isinstance(module, torch.nn.MaxPool2d):

            out = f"data/output/{clean}_output.bin"

            layers.append({
                "name": clean,
                "type": "maxpool",
                "input": prev_output,
                "output": out,
                "kernel_size": module.kernel_size,
                "stride": module.stride
            })

            prev_output = out

        # ================= LINEAR =================
        elif isinstance(module, torch.nn.Linear):

            out = f"data/output/{clean}_output.bin"

            layers.append({
                "name": clean,
                "type": "fc",
                "input": prev_output,
                "output": out,
                "weights": f"data/weights/{clean}_weight.bin",
                "bias": f"data/weights/{clean}_bias.bin",
                "in_features": module.in_features,
                "out_features": module.out_features
            })

            prev_output = out

        # ================= SKIP 1 =================
        if clean == "maxpool2":

            # Skip1 branch conv
            skip1_conv_out = "data/output/skip1_conv_output.bin"

            layers.append({
                "name": "skip1_conv",
                "type": "conv",
                "input": identity1,
                "output": skip1_conv_out,
                "weights": "data/weights/skip1_0_weight.bin",
                "bias": "data/weights/skip1_0_bias.bin",
                "kernel_size": 1,
                "stride": 4,
                "padding": 0,
                "in_channels": 32,
                "out_channels": 128
            })

            # ADD
            add_out = "data/output/skip1_add_output.bin"

            layers.append({
                "name": "skip1_add",
                "type": "add",
                "input1": prev_output,
                "input2": skip1_conv_out,
                "output": add_out
            })

            prev_output = add_out

            identity2 = prev_output  # for skip2

        # ================= SKIP 2 =================
        if clean == "maxpool3":

            skip2_conv_out = "data/output/skip2_conv_output.bin"

            layers.append({
                "name": "skip2_conv",
                "type": "conv",
                "input": identity2,
                "output": skip2_conv_out,
                "weights": "data/weights/skip2_0_weight.bin",
                "bias": "data/weights/skip2_0_bias.bin",
                "kernel_size": 1,
                "stride": 2,
                "padding": 0,
                "in_channels": 128,
                "out_channels": 256
            })

            add_out = "data/output/skip2_add_output.bin"

            layers.append({
                "name": "skip2_add",
                "type": "add",
                "input1": prev_output,
                "input2": skip2_conv_out,
                "output": add_out
            })

            prev_output = add_out

    # SAVE CONFIG
    with open(CONFIG_PATH, "w") as f:
        json.dump({"layers": layers}, f, indent=4)

    print("\n FULL config with skip connections generated!")


if __name__ == "__main__":
    main()
