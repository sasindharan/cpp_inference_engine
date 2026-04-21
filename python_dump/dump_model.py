import torch
import numpy as np
import os
from model import ResNet

BASE_DIR = "../data"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)

def save_bin(tensor, path):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(path)

    with open(path.replace(".bin", "_shape.txt"), "w") as f:
        f.write(str(list(arr.shape)))


layer_inputs = {}
layer_outputs = {}
feature_before_fc = None

def get_hook(name):
    def hook(module, input, output):
        try:
            if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
                layer_inputs[name] = input[0]

            if isinstance(output, torch.Tensor):
                layer_outputs[name] = output
                print(f"Captured: {name}")

        except Exception as e:
            print(f"Hook error at {name}: {e}")

    return hook

def fc_input_hook(module, input, output):
    global feature_before_fc
    if isinstance(input, tuple) and len(input) > 0:
        feature_before_fc = input[0]

def main():
    print("Starting model dump...")

    layer_inputs.clear()
    layer_outputs.clear()

    model = ResNet()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    print("Model loaded")

    # Register hooks for layers
    for name, layer in model.named_modules():
        if isinstance(layer, (
            torch.nn.Conv2d,
            torch.nn.BatchNorm2d,
            torch.nn.MaxPool2d,
            torch.nn.Linear
        )):
            layer.register_forward_hook(get_hook(name))

    # Hook to capture block4 (input to FC)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(fc_input_hook)
            print(f"FC hook registered on: {name}")
            break

    print("Hooks registered")

    # Input
    input_path = os.path.join(INPUT_DIR, "input.bin")

    if not os.path.exists(input_path):
        x = torch.randn(1, 3, 32, 32)
    else:
        x = torch.from_numpy(
            np.fromfile(input_path, dtype=np.float32)
        ).reshape(1, 3, 32, 32)

    save_bin(x, input_path)
    print("Input saved")

    # Forward
    print("Running forward pass...")
    with torch.no_grad():
        output = model(x)
    print("Forward completed")

    # Save final output
    save_bin(output, os.path.join(OUTPUT_DIR, "final_output.bin"))

    # Save block4 (input to FC)
    if feature_before_fc is not None:
        save_bin(
            feature_before_fc,
            os.path.join(OUTPUT_DIR, "block4_output.bin")
        )
        print("Saved block4_output.bin")

    # Save intermediate tensors
    for name in layer_outputs:
        clean_name = name.replace(".", "_")

        if name in layer_inputs:
            save_bin(
                layer_inputs[name],
                os.path.join(INPUT_DIR, f"{clean_name}_input.bin")
            )

        save_bin(
            layer_outputs[name],
            os.path.join(OUTPUT_DIR, f"{clean_name}_output.bin")
        )

    print("Intermediate tensors saved")

    # Save BatchNorm stats
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            clean_name = name.replace(".", "_")

            save_bin(
                module.running_mean,
                os.path.join(WEIGHT_DIR, f"{clean_name}_running_mean.bin")
            )

            save_bin(
                module.running_var,
                os.path.join(WEIGHT_DIR, f"{clean_name}_running_var.bin")
            )

    print("BatchNorm weights saved")

    # Save FC weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            clean = name.replace(".", "_")

            save_bin(
                module.weight,
                os.path.join(WEIGHT_DIR, f"{clean}_weight.bin")
            )

            save_bin(
                module.bias,
                os.path.join(WEIGHT_DIR, f"{clean}_bias.bin")
            )

    print("FC weights saved")

    # CORRECT FIX: Save ReLU(FC1)
    fc1_output = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name in layer_outputs:
                fc1_output = layer_outputs[name]
                print(f"Using FC1 layer: {name}")
                break

    if fc1_output is None:
        print("ERROR: FC1 output not found!")
    else:
        relu_fc1 = torch.relu(fc1_output)

        save_bin(
            relu_fc1,
            os.path.join(OUTPUT_DIR, "relu_fc1_output.bin")
        )

        print("Saved relu_fc1_output.bin")

    print("\nMODEL DUMP COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
