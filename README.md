<div align="center"><img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy : NumPy & SciPy for GPU

***

**NOTE**:

This is a special set of commits for applications developed by the AMD DC GPU AISS team.

***

[![pypi](https://img.shields.io/pypi/v/cupy)](https://pypi.python.org/pypi/cupy)
[![Conda](https://img.shields.io/conda/vn/conda-forge/cupy)](https://anaconda.org/conda-forge/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy)](https://github.com/cupy/cupy)
[![Gitter](https://badges.gitter.im/cupy/community.svg)](https://gitter.im/cupy/community)
[![Twitter](https://img.shields.io/twitter/follow/CuPy_Team?label=%40CuPy_Team)](https://twitter.com/CuPy_Team)

[**Website**](https://cupy.dev/)
| [**Install**](https://docs.cupy.dev/en/stable/install.html)
| [**Tutorial**](https://docs.cupy.dev/en/stable/user_guide/basic.html)
| [**Examples**](https://github.com/cupy/cupy/tree/main/examples)
| [**Documentation**](https://docs.cupy.dev/en/stable/)
| [**API Reference**](https://docs.cupy.dev/en/stable/reference/)
| [**Forum**](https://groups.google.com/forum/#!forum/cupy)

CuPy is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python.
CuPy acts as a [drop-in replacement](https://docs.cupy.dev/en/stable/reference/comparison.html) to run existing NumPy/SciPy code on NVIDIA CUDA or AMD ROCm platforms.

```py
>>> import cupy as cp
>>> x = cp.arange(6).reshape(2, 3).astype('f')
>>> x
array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.]], dtype=float32)
>>> x.sum(axis=1)
array([  3.,  12.], dtype=float32)
```

CuPy also provides access to low-level CUDA features.
You can pass `ndarray` to existing CUDA C/C++ programs via [RawKernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html#raw-kernels), use [Streams](https://docs.cupy.dev/en/stable/reference/cuda.html) for performance, or even call [CUDA Runtime APIs](https://docs.cupy.dev/en/stable/reference/cuda.html#runtime-api) directly.

## Install from Source

***

**NOTE (Tested ROCm versions):**

The following installation instructions have been tested with ROCm 6.0.0 ... ROCm 6.2.0 (inclusive).
They may or may not work with more recent or older releases of ROCm.

***

We create our build environment via `conda` [^miniconda]. The environment file (`cupy.yaml`) is shown below.
Note that while you can adapt the Python version, the Cython version should stay fixed.

```yaml
# file: cupy.yaml
channels:
- conda-forge
dependencies:
- python==3.10.13 # adapt to your needs
- cython==0.29.35
```

We then build as follows:

```bash
# initialize conda environment
conda env create -n cupy_dev -f cupy_dev.yaml
conda activate cupy_dev # now we are working in the `cupy_dev` conda env

pip install --upgrade pip # always recommended

# cd <path/to/parent-directory>
git clone https://github.com/ROCm/cupy.git -b aiss/cai-branch
cd cupy
git submodule update --init
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm        # adapt to your environment
export HCC_AMDGPU_TARGET="gfx90a" # adapt to your AMD GPU architecture
python3 setup.py bdist_wheel      # build the wheel
pip install dist/cupy*.whl        # install the wheel
```

[^miniconda]: <https://docs.anaconda.com/miniconda/#>

## License

MIT License (see `LICENSE` file).

CuPy is designed based on NumPy's API and SciPy's API (see `docs/LICENSE_THIRD_PARTY` file).

CuPy is being developed and maintained by [Preferred Networks](https://www.preferred.jp/en/) and [community contributors](https://github.com/cupy/cupy/graphs/contributors).

## Reference

Ryosuke Okuta, Yuya Unno, Daisuke Nishino, Shohei Hido and Crissman Loomis.
**CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations.**
*Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*, (2017).
[[PDF](http://learningsys.org/nips17/assets/papers/paper_16.pdf)]

```bibtex
@inproceedings{cupy_learningsys2017,
  author       = "Okuta, Ryosuke and Unno, Yuya and Nishino, Daisuke and Hido, Shohei and Loomis, Crissman",
  title        = "CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations",
  booktitle    = "Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)",
  year         = "2017",
  url          = "http://learningsys.org/nips17/assets/papers/paper_16.pdf"
}
```
