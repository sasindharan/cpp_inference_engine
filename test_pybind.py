import numpy as np
import cpp_engine

print("Running test...")

x = np.fromfile("data/input/input.bin", dtype=np.float32)

out = cpp_engine.run_model(x)

ref = np.fromfile("data/output/final_output.bin", dtype=np.float32)

print("Max diff:", np.abs(out - ref).max())

assert np.allclose(out, ref, atol=1e-4)

print("PyBind WORKS")