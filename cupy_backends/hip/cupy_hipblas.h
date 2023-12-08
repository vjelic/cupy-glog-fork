#ifndef INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H
#define INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H

#include "cupy_hip_common.h"
#if HIP_VERSION >= 50530600 
#include <hipblas/hipblas.h>
#else
#include <hipblas.h>
#endif
#include <hip/hip_version.h>  // for HIP_VERSION
#include <stdexcept>  // for gcc 10


extern "C" {

///////////////////////////////////////////////////////////////////////////////
// blas & lapack (hipBLAS/rocBLAS & rocSOLVER)
///////////////////////////////////////////////////////////////////////////////

/* As of ROCm 3.5.0 (this may have started earlier) many rocSOLVER helper functions
 * are deprecated and using their counterparts from rocBLAS is recommended. In
 * particular, rocSOLVER simply uses rocBLAS's handle for its API calls. This means
 * they are much more integrated than cuBLAS and cuSOLVER are, so it is better to
 * put all of the relevant function in one place.
 */

// TODO(leofang): investigate if we should just remove the hipBLAS layer and use
// rocBLAS directly, since we need to expose its handle anyway


/* ---------- helpers ---------- */
static hipblasOperation_t convert_hipblasOperation_t(cublasOperation_t op) {
    return static_cast<hipblasOperation_t>(static_cast<int>(op) + 111);
}

static hipblasFillMode_t convert_hipblasFillMode_t(cublasFillMode_t mode) {
    switch(static_cast<int>(mode)) {
        case 0 /* CUBLAS_FILL_MODE_LOWER */: return HIPBLAS_FILL_MODE_LOWER;
        case 1 /* CUBLAS_FILL_MODE_UPPER */: return HIPBLAS_FILL_MODE_UPPER;
        default: throw std::runtime_error("unrecognized mode");
    }
}

static hipblasDiagType_t convert_hipblasDiagType_t(cublasDiagType_t type) {
    return static_cast<hipblasDiagType_t>(static_cast<int>(type) + 131);
}

static hipblasSideMode_t convert_hipblasSideMode_t(cublasSideMode_t mode) {
    return static_cast<hipblasSideMode_t>(static_cast<int>(mode) + 141);
}

static hipblasDatatype_t convert_hipblasDatatype_t(cudaDataType_t type) {
    switch(static_cast<int>(type)) {
        case 0 /* CUDA_R_32F */: return HIPBLAS_R_32F;
        case 1 /* CUDA_R_64F */: return HIPBLAS_R_64F;
        case 2 /* CUDA_R_16F */: return HIPBLAS_R_16F;
        case 3 /* CUDA_R_8I */ : return HIPBLAS_R_8I;
        case 4 /* CUDA_C_32F */: return HIPBLAS_C_32F;
        case 5 /* CUDA_C_64F */: return HIPBLAS_C_64F;
        case 6 /* CUDA_C_16F */: return HIPBLAS_C_16F;
        case 7 /* CUDA_C_8I */ : return HIPBLAS_C_8I;
        case 8 /* CUDA_R_8U */ : return HIPBLAS_R_8U;
        case 9 /* CUDA_C_8U */ : return HIPBLAS_C_8U;
        default: throw std::runtime_error("unrecognized type");
    }
}


// Context
hipblasStatus_t cublasGetVersion(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// Math Mode
hipblasStatus_t cublasSetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t cublasGetMathMode(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// BLAS Level 3
hipblasStatus_t cublasSgemmEx(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t cublasGemmEx_v11(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t cublasGemmStridedBatchedEx_v11(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// BLAS extension
hipblasStatus_t cublasStrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t cublasDtrttp(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t cublasStpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t cublasDtpttr(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

#if HIP_VERSION < 308
hipblasStatus_t hipblasSgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetriBatched(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
#elif HIP_VERSION < 307
hipblasStatus_t hipblasCgeam(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeam(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
#elif HIP_VERSION < 306
hipblasStatus_t hipblasSdgmm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDdgmm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdgmm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdgmm(...) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
#endif

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPBLAS_H
