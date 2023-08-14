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

extern "C" {

#define CUSPARSE_VERSION (hipsparseVersionMajor*100000+hipsparseVersionMinor*100+hipsparseVersionPatch)

typedef hipsparseIndexBase_t cusparseIndexBase_t;
typedef hipsparseMatrixType_t cusparseMatrixType_t;
typedef hipsparsePointerMode_t cusparsePointerMode_t;
typedef hipsparseAction_t cusparseAction_t;
typedef enum {} cusparseAlgMode_t;

#if HIP_VERSION < 50631061
typedef enum {} cusparseCsr2CscAlg_t;
#else
typedef hipsparseCsr2CscAlg_t cusparseCsr2CscAlg_t;
#endif
typedef hipsparseSpGEMMAlg_t cusparseSpGEMMAlg_t;

#if HIP_VERSION >= 402
typedef hipsparseSpVecDescr_t cusparseSpVecDescr_t;
#else
typedef void* cusparseSpVecDescr_t;
#endif
#if HIP_VERSION >= 402
typedef hipsparseFormat_t cusparseFormat_t;
#else
typedef enum {} cusparseFormat_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseOrder_t cusparseOrder_t;
#else
typedef enum {} cusparseOrder_t;
#endif

#if HIP_VERSION >= 50000000
typedef hipsparseSpMatAttribute_t cusparseSpMatAttribute_t;
typedef hipsparseSpSMAlg_t cusparseSpSMAlg_t;
typedef hipsparseSpSMDescr_t cusparseSpSMDescr_t;
#else
typedef enum {} cusparseSpMatAttribute_t;
typedef enum {} cusparseSpSMAlg_t;
typedef void * cusparseSpSMDescr_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseSparseToDenseAlg_t cusparseSparseToDenseAlg_t;
#else
typedef enum {} cusparseSparseToDenseAlg_t;
#endif

#if HIP_VERSION >= 402
typedef hipsparseDenseToSparseAlg_t cusparseDenseToSparseAlg_t;
#else
typedef enum {} cusparseDenseToSparseAlg_t;
#endif

#if HIP_VERSION < 50631061
hipsparseStatus_t hipsparseSgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMatGetStridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMatSetStridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDnMatGetStridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDnMatSetStridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCsr2cscEx2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

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

hipsparseStatus_t cusparseGather(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

}  // extern "C"


#endif  // INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
