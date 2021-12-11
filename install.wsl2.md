This article is a guide for one who wants to run and build CUDA applications in WSL.

----

### Install GPU Drivers on Windows

Follow this [article](https://developer.nvidia.com/cuda/wsl).



### Install CUDA on WSL

1. Install CUDA and Kits

   Follow this [article](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch03a-setting-up-cuda) with step 4.2.6 ~ 4.2.7, or run the command below in WSL:

   **Note:** The software versions are examples only. In the command below, use the CUDA and driver versions that you want to install. Also take care about [NVIDIA Compute Software Support Matrix for WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#support-matrix-for-wsl2).

   ```bash
   # install cuda
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin && sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
   sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
   sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
   sudo apt-get update
   sudo apt-get -y install cuda
   
   # install development kits
   apt-get install -y cuda-toolkit-11-4
   ```

2. Set environment variables

   Edit your **.*YOUR_BASH*rc**(eg: `.bashrc`, `.zshrc` etc.)

   ```bash
   # To use binary globally(eg: nvcc)
   export PATH="/usr/local/cuda/bin:$PATH"
   # Set global library including CUDA library
   export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
   ```

   Then, don't forget using `source` command.

3. Check

   ```bash
   ❯ nvcc --version
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2021 NVIDIA Corporation
   Built on Wed_Jun__2_19:15:15_PDT_2021
   Cuda compilation tools, release 11.4, V11.4.48
   Build cuda_11.4.r11.4/compiler.30033411_0
   ```
