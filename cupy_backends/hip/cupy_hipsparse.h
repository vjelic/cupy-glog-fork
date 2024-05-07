#ifndef INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
#define INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
#if HIP_VERSION >= 50530600
#include <hipsparse/hipsparse.h>
#else
#include <hipsparse.h>
#endif
#include <hip/hip_version.h>    // for HIP_VERSION
#include <hip/library_types.h>  // for hipDataType
#include <stdexcept>  // for gcc 10.0

typedef enum {} cusparseAlgMode_t;

hipsparseStatus_t cusparseCsrmvEx_bufferSize(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCsrmvEx(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseConstrainedGeMM_bufferSize(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseConstrainedGeMM(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

#define CUSPARSE_VERSION (hipsparseVersionMajor*100000+hipsparseVersionMinor*100+hipsparseVersionPatch)

// cuSPARSE generic API
#if HIP_VERSION < 402
typedef void* cusparseSpVecDescr_t;
typedef void* cusparseDnVecDescr_t;
typedef void* cusparseSpMatDescr_t;
typedef void* cusparseDnMatDescr_t;
typedef enum {} cusparseIndexType_t;
typedef enum {} cusparseFormat_t;
typedef enum {} cusparseOrder_t;
typedef enum {} cusparseSpMVAlg_t;
typedef enum {} cusparseSpMMAlg_t;
typedef enum {} cusparseSparseToDenseAlg_t;
typedef enum {} cusparseDenseToSparseAlg_t;
#endif

#if HIP_VERSION < 50000000
typedef enum {} cusparseSpMatAttribute_t;
typedef enum {} cusparseSpSMAlg_t;
typedef void * cusparseSpSMDescr_t;
#endif



#endif  // INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
