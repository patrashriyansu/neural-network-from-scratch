"""Generate environment report for results/environment_report.txt"""
import sys
import platform
import datetime
import numpy as np
import torch
import sklearn
import matplotlib

lines = []
lines.append("Environment Report")
lines.append("=" * 50)
now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines.append(f"Generated     : {now_str}")
lines.append(f"OS            : {platform.system()} {platform.release()} ({platform.machine()})")
lines.append(f"Python        : {sys.version}")
lines.append("")
lines.append("Package Versions")
lines.append("-" * 50)
lines.append(f"NumPy         : {np.__version__}")
lines.append(f"PyTorch       : {torch.__version__}")
lines.append(f"scikit-learn  : {sklearn.__version__}")
lines.append(f"Matplotlib    : {matplotlib.__version__}")
lines.append("")
lines.append("PyTorch / CUDA")
lines.append("-" * 50)
lines.append(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    lines.append(f"GPU Name      : {torch.cuda.get_device_name(0)}")
    lines.append(f"CUDA version  : {torch.version.cuda}")
else:
    lines.append("GPU           : Not available (CPU mode)")
    lines.append("CPU fallback  : Active")
lines.append("")
lines.append("Reproducibility")
lines.append("-" * 50)
lines.append("NumPy seed    : 42")
lines.append("PyTorch seed  : 42")
lines.append("sklearn split : random_state=42, stratified=True")

report = "\n".join(lines)
print(report)
with open("results/environment_report.txt", "w") as f:
    f.write(report)
print("\nSaved to results/environment_report.txt")
