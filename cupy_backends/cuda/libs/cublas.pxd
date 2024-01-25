"""Thin wrapper of CUBLAS."""
from libc.stdint cimport intptr_t


###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef void* cuComplexPtr 'cuComplex*'
    ctypedef void* cuDoubleComplexPtr 'cuDoubleComplex*'


cdef extern from *:
    ctypedef void* Handle 'cublasHandle_t'

    ctypedef int DiagType 'cublasDiagType_t'
    ctypedef int FillMode 'cublasFillMode_t'
    ctypedef int Operation 'cublasOperation_t'
    ctypedef int PointerMode 'cublasPointerMode_t'
    ctypedef int SideMode 'cublasSideMode_t'
    ctypedef int GemmAlgo 'cublasGemmAlgo_t'
    ctypedef int Math 'cublasMath_t'
    ctypedef int ComputeType 'cublasComputeType_t'


###############################################################################
# Enum
###############################################################################
IF CUPY_HIP_VERSION != 0:
    cpdef enum:
        # need to revisit this when cython supports C++ enums (in 3.0)
        # https://stackoverflow.com/a/67138945
        CUBLAS_OP_N = 111
        CUBLAS_OP_T = 112
        CUBLAS_OP_C = 113

        CUBLAS_POINTER_MODE_HOST = 0
        CUBLAS_POINTER_MODE_DEVICE = 1

        CUBLAS_SIDE_LEFT = 141
        CUBLAS_SIDE_RIGHT = 142

        CUBLAS_FILL_MODE_LOWER = 122
        CUBLAS_FILL_MODE_UPPER = 121

        CUBLAS_DIAG_NON_UNIT = 131
        CUBLAS_DIAG_UNIT = 132

        CUBLAS_GEMM_DEFAULT = 160
        CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99

        # The following two are left for backward compatibility; renamed from
        # `DFALT` to `DEFAULT` in CUDA 9.1.
        CUBLAS_GEMM_DFALT = 160
        CUBLAS_GEMM_DFALT_TENSOR_OP = 99

        CUBLAS_DEFAULT_MATH = 0
        CUBLAS_TENSOR_OP_MATH = 1

        CUBLAS_COMPUTE_16F = 0             # half - default
        CUBLAS_COMPUTE_16F_PEDANTIC = 1    # half - pedantic
        CUBLAS_COMPUTE_32F = 2             # float - default
        CUBLAS_COMPUTE_32F_PEDANTIC = 3    # float - pedantic
        CUBLAS_COMPUTE_32F_FAST_16F = 4    # float - fast, allows down-converting inputs to half or TF32  # NOQA
        CUBLAS_COMPUTE_32F_FAST_16BF = 5   # float - fast, allows down-converting inputs to bfloat16 or TF32  # NOQA
        CUBLAS_COMPUTE_32F_FAST_TF32 = 6   # float - fast, allows down-converting inputs to TF32  # NOQA
        CUBLAS_COMPUTE_64F = 7             # double - default
        CUBLAS_COMPUTE_64F_PEDANTIC = 8    # double - pedantic
        CUBLAS_COMPUTE_32I = 9             # signed 32-bit int - default
        CUBLAS_COMPUTE_32I_PEDANTIC = 10   # signed 32-bit int - pedantic
ELSE:
    cpdef enum:
        # need to revisit this when cython supports C++ enums (in 3.0)
        # https://stackoverflow.com/a/67138945
        CUBLAS_OP_N = 0
        CUBLAS_OP_T = 1
        CUBLAS_OP_C = 2

        CUBLAS_POINTER_MODE_HOST = 0
        CUBLAS_POINTER_MODE_DEVICE = 1

        CUBLAS_SIDE_LEFT = 0
        CUBLAS_SIDE_RIGHT = 1

        CUBLAS_FILL_MODE_LOWER = 0
        CUBLAS_FILL_MODE_UPPER = 1

        CUBLAS_DIAG_NON_UNIT = 0
        CUBLAS_DIAG_UNIT = 1

        CUBLAS_GEMM_DEFAULT = -1
        CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99

        # The following two are left for backward compatibility; renamed from
        # `DFALT` to `DEFAULT` in CUDA 9.1.
        CUBLAS_GEMM_DFALT = -1
        CUBLAS_GEMM_DFALT_TENSOR_OP = 99

        CUBLAS_DEFAULT_MATH = 0
        CUBLAS_TENSOR_OP_MATH = 1

        # cublasComputeType_t added in CUDA 11.0
        CUBLAS_COMPUTE_16F = 64            # half - default
        CUBLAS_COMPUTE_16F_PEDANTIC = 65   # half - pedantic
        CUBLAS_COMPUTE_32F = 68            # float - default
        CUBLAS_COMPUTE_32F_PEDANTIC = 69   # float - pedantic
        CUBLAS_COMPUTE_32F_FAST_16F = 74   # float - fast, allows down-converting inputs to half or TF32  # NOQA
        CUBLAS_COMPUTE_32F_FAST_16BF = 75  # float - fast, allows down-converting inputs to bfloat16 or TF32  # NOQA
        CUBLAS_COMPUTE_32F_FAST_TF32 = 77  # float - fast, allows down-converting inputs to TF32  # NOQA
        CUBLAS_COMPUTE_64F = 70            # double - default
        CUBLAS_COMPUTE_64F_PEDANTIC = 71   # double - pedantic
        CUBLAS_COMPUTE_32I = 72            # signed 32-bit int - default
        CUBLAS_COMPUTE_32I_PEDANTIC = 73   # signed 32-bit int - pedantic

