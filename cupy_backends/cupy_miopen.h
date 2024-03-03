#ifndef INCLUDE_GUARD_CUPY_MIOPEN_H
#define INCLUDE_GUARD_CUPY_MIOPEN_H

#if CUPY_USE_HIP

#include "hip/cupy_miopen.h"

#else // #ifndef CUPY_USE_HIP

#include "stub/cupy_miopen.h"

#endif // #ifndef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_MIOPEN_H
