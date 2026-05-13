# paste this into a quick debug script: check_dx.py
import numpy as np

dx_path = "apbs_debug/4g5j_pot.dx" 
with open(dx_path) as f:
    lines = f.readlines()

# Print the header
print("=== HEADER (first 20 lines) ===")
for i, l in enumerate(lines[:20]):
    print(f"{i:3d}: {l}", end="")

# Find data start and print stats
data_vals = []
in_data = False
for line in lines:
    if line.strip().startswith("object 3"):
        in_data = True
        continue
    if in_data:
        if line.strip().startswith("object") or line.strip().startswith("attribute"):
            break
        data_vals.extend(float(v) for v in line.split())

arr = np.array(data_vals)
print(f"\n=== DATA STATS ===")
print(f"  N values: {len(arr)}")
print(f"  Min: {arr.min():.4f}")
print(f"  Max: {arr.max():.4f}")
print(f"  Mean: {arr.mean():.4f}")
print(f"  Std: {arr.std():.4f}")
print(f"  Values near zero (|v|<1): {(np.abs(arr)<1).sum()}")
print(f"  Values |v|>100: {(np.abs(arr)>100).sum()}")