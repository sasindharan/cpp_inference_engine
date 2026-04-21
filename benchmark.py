import time
import numpy as np
import torch
from python_dump.model import ResNet
import cpp_engine

model = ResNet()
model.load_state_dict(torch.load("python_dump/best_model.pth", map_location="cpu"))
model.eval()

x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_flat = x.flatten()

# C++
start = time.time()
for _ in range(100):
    cpp_engine.run_model(x_flat)
print("C++ time:", time.time() - start)

# PyTorch
xt = torch.tensor(x)

start = time.time()
for _ in range(100):
    model(xt)
print("PyTorch time:", time.time() - start)