#ifndef INCLUDE_GUARD_CUDA_CUPY_CUDNN_H
#define INCLUDE_GUARD_CUDA_CUPY_CUDNN_H

// TODO: Do we need CUPY_USE_HIP here, if this file is only used on HIP path
#if CUPY_USE_HIP

#include "miopen/miopen.h"

#endif // #ifdef CUPY_USE_HIP

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUDNN_H
