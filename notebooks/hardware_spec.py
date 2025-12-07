# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import psutil
import platform

# 1. Get CPU Details
print(f"Processor: {platform.processor()}")
print(f"Machine Type: {platform.machine()}")
print(f"System: {platform.system()} {platform.release()}")

# 2. Get RAM Details
ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
print(f"Total RAM: {ram_gb} GB")

# 3. Get Specific CPU Model (Linux/Mac/Windows specific tricks)
try:
    # This works best on Linux/Windows to get the specific model name (e.g., i7-12700H)
    import cpuinfo
    info = cpuinfo.get_cpu_info()
    print(f"CPU Model: {info['brand_raw']}")
except ImportError:
    print("To get the exact CPU model name, run: pip install py-cpuinfo")

# %%
