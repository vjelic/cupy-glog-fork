#!/bin/bash

set -uex

apt-get -y update
DEBIAN_FRONTEND=noninteractive apt-get -y install python3.9-dev python3-pip git

hipconfig

python3.9 -m pip install -U pip wheel
pip install git+https://github.com/ROCmSoftwarePlatform/hipify_torch.git

export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx900"
export CUPY_INSTALL_USE_HIP="1"
pip3 install -v -e .
python3 -c "import cupy; cupy.show_config()"
