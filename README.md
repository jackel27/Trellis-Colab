# TRELLIS Colab Setup
---
Trellis colab notebook. jupyter. whatever. working setup!
This README details the environment setup, package versions, and steps used to run the [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) project, along with the [mip-splatting](https://github.com/autonomousvision/mip-splatting) submodules, **FlashAttention** (2.5.8), **spconv**, and **Kaolin** (0.17.0). Everything is pinned to **CUDA 11.8** and **PyTorch 2.3.0** for reproducibility.

### If you are looking for just the Google Colab - go here: [Kaiser-Trellis.ipynb](https://github.com/jackel27/Trellis-Colab/edit/main/Kaiser-Trellis.ipynb)





## Table of Contents
1. [Runtime Requirements](#runtime-requirements)  
2. [Dependencies & Versions](#dependencies--versions)  
3. [Steps to Reproduce in Google Colab](#steps-to-reproduce-in-google-colab)  
    1. [Install CUDA 11.8](#1-install-cuda-118)  
    2. [Create Conda Environment (`trellis`)](#2-create-conda-environment-trellis)  
    3. [Install System Dev Tools](#3-install-system-dev-tools)  
    4. [Clone & Setup TRELLIS](#4-clone--setup-trellis)  
    5. [Clone & Setup mip-splatting](#5-clone--setup-mip-splatting)  
    6. [Launching TRELLIS with Ngrok](#6-launching-trellis-with-ngrok)  
4. [References & Links](#references--links)

---

## Runtime Requirements

- **NVIDIA GPU** with CUDA Compute Capability >= 7.0 (recommended)  
- **Google Colab GPU Runtime** (or equivalent local environment)
- **Ubuntu 22.04** or a similar Linux environment for best compatibility
- **Miniconda/Conda** environment management

When using Colab, be sure to enable a GPU runtime:  
> **Runtime** > **Change runtime type** > **Hardware accelerator**: **GPU**.

---

## Dependencies & Versions

Below is a summary of major packages, pinned versions, and the sources we use:

1. **Operating System**  
   - Ubuntu 22.04 (default Google Colab base image)

2. **CUDA**  
   - CUDA **11.8** (installed from NVIDIA’s Ubuntu 22.04 repo)  
   - GPG key and repo from:  
     `[https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/)`

3. **Python**  
   - Python **3.11** (via Miniconda)

4. **Conda Environment**  
   - Environment name: `trellis`

5. **PyTorch & Related**  
   - PyTorch **2.3.0**  
   - TorchVision **0.18.0**  
   - TorchAudio **2.3.0**  
   - `pytorch-cuda=11.8`

6. **FlashAttention**  
   - `flash-attn==2.5.8` (cu118, torch2.3, cxx11abiFALSE build) from  
     [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/releases)

7. **spconv**  
   - `spconv-cu118` (PyPI)

8. **Kaolin**  
   - `kaolin==0.17.0`, installed from  
     [https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu118.html](https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu118.html)

9. **Pillow**  
   - `pillow<11.0` (forced reinstall to avoid compatibility issues)

10. **Gradio**  
    - `gradio==4.44.1`  
    - `gradio_litmodel3d==0.0.1`

11. **System Tools**  
    - `build-essential`, `cmake`, `ninja-build`, `nvidia-cuda-toolkit`  
    - Python build libraries: `pip`, `setuptools`, `wheel`, `ninja`, `Cython`

12. **Microsoft TRELLIS**  
    - Cloned from [https://github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) with submodules  
    - Submodule builds: `flash-attn`, `spconv`, `mipgaussian`, `nvdiffrast`, `kaolin`  
    - Patched `app.py` to use `demo.launch(share=True)`

13. **mip-splatting**  
    - Cloned from [https://github.com/autonomousvision/mip-splatting](https://github.com/autonomousvision/mip-splatting)  
    - Installs the `diff_gaussian_rasterization` submodule

14. **Ngrok**  
    - Used to expose the local Gradio app externally  
    - Make sure you have an **ngrok auth token**

---

## Steps to Reproduce in Google Colab 
(or just view the kaiser-colab file for full cells)

### 1. Install CUDA 11.8

1. Update apt and install `gnupg`
2. Download and install `cuda-keyring_1.1-1_all.deb` from NVIDIA
3. `apt-get update`
4. Install `cuda-11-8`
5. (Optional) Check `nvcc --version`

```bash
# Example snippet:
sudo apt-get update -y
sudo apt-get install -y gnupg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y
sudo apt-get install -y cuda-11-8
export PATH="/usr/local/cuda-11.8/bin:$PATH"
nvcc --version
```

### 2. Create Conda Environment (`trellis`)

1. Install Miniconda inside Colab using `condacolab`
2. Activate conda environment in a Bash cell
3. Create `trellis` with Python 3.11
4. Install PyTorch 2.3.0 + CUDA 11.8, flash-attn, spconv, kaolin, etc.

```bash
# Example snippet:
pip install -q condacolab
python -c "import condacolab;condacolab.install_miniconda()"

# Then in a bash cell:
source /usr/local/etc/profile.d/conda.sh
conda create -y -n trellis python=3.11
conda activate trellis
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install spconv-cu118
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu118.html
pip install --force-reinstall "pillow<11.0"
pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
```

### 3. Install System Dev Tools

1. Install build-essential, cmake, ninja-build, nvidia-cuda-toolkit
2. Activate `trellis` again
3. Upgrade Python build libraries

```bash
apt-get update -y
apt-get install -y build-essential cmake ninja-build nvidia-cuda-toolkit
source /usr/local/etc/profile.d/conda.sh
conda activate trellis
pip install --upgrade pip setuptools wheel ninja Cython
```

### 4. Clone & Setup TRELLIS

1. Remove any previous `/content/TRELLIS`
2. `git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /content/TRELLIS`
3. `cd /content/TRELLIS && git submodule update --init --recursive`
4. Create `assets/example_image` so `app.py` doesn’t crash
5. Run `setup.sh --basic --flash-attn --spconv --mipgaussian --nvdiffrast --kaolin --demo`
6. Patch `app.py` to set `demo.launch(share=True)`

```bash
rm -rf /content/TRELLIS
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /content/TRELLIS
cd /content/TRELLIS
git submodule update --init --recursive
mkdir -p /content/TRELLIS/assets/example_image
bash ./setup.sh --basic --flash-attn --spconv --mipgaussian --nvdiffrast --kaolin --demo
sed -i 's/demo.launch()$/demo.launch(share=True)/' app.py
```

### 5. Clone & Setup mip-splatting

1. `git clone --recurse-submodules https://github.com/autonomousvision/mip-splatting.git`
2. `cd mip-splatting && git submodule update --init --recursive`
3. Install system libs: `libgl1-mesa-dev`, etc.
4. Install `diff_gaussian_rasterization` submodule

```bash
cd /content
git clone --recurse-submodules https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting
git submodule update --init --recursive
apt-get install -y libgl1-mesa-dev
cd submodules/diff-gaussian-rasterization
pip install . --no-build-isolation -v
```

### 6. Launching TRELLIS with Ngrok

1. Make sure `ngrok` is installed in your environment or is available on Colab  
2. Set your **ngrok auth token** (replace `YOUR_NGROK_AUTH_TOKEN` below)  
3. Expose port 7860  
4. Run `app.py` in the `trellis` environment

```python
import subprocess, time, json

NGROK_AUTH = "YOUR_NGROK_AUTH_TOKEN"

subprocess.run(["ngrok", "config", "add-authtoken", NGROK_AUTH], check=True)
proc_ngrok = subprocess.Popen(["ngrok", "http", "7860"])

time.sleep(4)
tunnels_json = subprocess.check_output(["curl", "-s", "127.0.0.1:4040/api/tunnels"])
public_url = json.loads(tunnels_json)["tunnels"][0]["public_url"]
print("NGROK URL =>", public_url)

p = subprocess.Popen(
    ["conda", "run", "-n", "trellis", "python", "-u", "app.py"],
    cwd="/content/TRELLIS",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)

try:
    for line in iter(p.stdout.readline, b""):
        if not line:
            break
        print(line.decode(), end="")
except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")
    p.kill()
finally:
    p.wait()
```

You will see logs from the Gradio/TRELLIS app in real-time. Click the `NGROK URL` to open the Gradio interface in your browser.

---

## References & Links

1. **TRELLIS (Microsoft)**  
   - GitHub: [https://github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)

2. **mip-splatting (autonomousvision)**  
   - GitHub: [https://github.com/autonomousvision/mip-splatting](https://github.com/autonomousvision/mip-splatting)

3. **FlashAttention**  
   - [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

4. **Kaolin (NVIDIA)**  
   - [https://github.com/NVIDIAGameWorks/kaolin/](https://github.com/NVIDIAGameWorks/kaolin/)  
   - Wheels: [https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu118.html](https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu118.html)

5. **Ngrok**  
   - [https://ngrok.com/](https://ngrok.com/)  
   - After you sign up, get your token from the [dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).

6. **condacolab**  
   - [https://github.com/conda-incubator/condacolab](https://github.com/conda-incubator/condacolab)

---

### Contributing

Feel free to open issues or pull requests if you encounter any build problems or want to add instructions for other OS distributions.

**Enjoy exploring TRELLIS, Mip-Splatting, and advanced PyTorch-based 3D frameworks!**
