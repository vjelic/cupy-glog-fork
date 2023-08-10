#!/bin/bash

set -uex

# Python 3.8 (Ubuntu 20.04)
apt-get -y update
DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip python3-dev git

hipconfig

pip3 install -U pip wheel

# install hipify_torch
pip3 install git+https://github.com/ROCmSoftwarePlatform/hipify_torch.git

export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx900"
export CUPY_INSTALL_USE_HIP="1"
pip3 install -v -e .
python3 -c "import cupy; cupy.show_config()"
