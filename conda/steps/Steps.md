
FROM:
https://github.com/conda-forge/miniforge

GET:
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe

```
# Update Mamba
mamba update -n base mamba

# Create environment
mamba create -n mla_202411 python=3.11.10

# Activate environment
mamba activate mla_202411

# Install from yml
mamba env update -n mla_202411 --file environment_fromhistory_v2.yml
```