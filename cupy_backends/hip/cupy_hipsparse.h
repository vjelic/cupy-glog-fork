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

#if HIP_VERSION < 401
#define HIPSPARSE_STATUS_NOT_SUPPORTED (hipsparseStatus_t)10
#endif

extern "C" {
#if HIP_VERSION < 308
typedef void* bsric02Info_t;
#endif

#if HIP_VERSION < 309
typedef void* bsrilu02Info_t;
#endif
typedef enum {} cusparseAlgMode_t;

#if HIP_VERSION < 600
// Error handling
const char* cusparseGetErrorName(...) {
    // Unavailable in hipSparse; this should not be called
    return "CUPY_HIPSPARSE_BINDING_UNEXPECTED_ERROR";
}
const char* cusparseGetErrorString(...) {
    // Unavailable in hipSparse; this should not be called
    return "unexpected error in CuPy hipSparse binding";
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
#if HIP_VERSION < 309
hipsparseStatus_t cusparseCreateBsrilu02Info(bsrilu02Info_t* info) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
hipsparseStatus_t cusparseDestroyBsrilu02Info(bsrilu02Info_t info) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
hipsparseStatus_t cusparseSbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                float*           boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                double*          boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuComplex*       boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZbsrilu02_numericBoost(cusparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuDoubleComplex* boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle,
                                             bsrilu02Info_t   info,
                                             int*             position) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSbsrilu02_bufferSize(cusparseHandle_t         handle,
                                              cusparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const cusparseMatDescr_t descrA,
                                              float*                   bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDbsrilu02_bufferSize(cusparseHandle_t         handle,
                                              cusparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const cusparseMatDescr_t descrA,
                                              double*                  bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCbsrilu02_bufferSize(cusparseHandle_t         handle,
                                              cusparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const cusparseMatDescr_t descrA,
                                              cuComplex*               bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZbsrilu02_bufferSize(cusparseHandle_t         handle,
                                              cusparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const cusparseMatDescr_t descrA,
                                              cuDoubleComplex*         bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSbsrilu02_analysis(cusparseHandle_t         handle,
                                            cusparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const cusparseMatDescr_t descrA,
                                            float*                   bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDbsrilu02_analysis(cusparseHandle_t         handle,
                                            cusparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const cusparseMatDescr_t descrA,
                                            double*                  bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCbsrilu02_analysis(cusparseHandle_t         handle,
                                            cusparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const cusparseMatDescr_t descrA,
                                            cuComplex*               bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZbsrilu02_analysis(cusparseHandle_t         handle,
                                            cusparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const cusparseMatDescr_t descrA,
                                            cuDoubleComplex*         bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            cusparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSbsrilu02(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   float*                   bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   cusparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDbsrilu02(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   double*                  bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   cusparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCbsrilu02(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   cuComplex*               bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   cusparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZbsrilu02(cusparseHandle_t         handle,
                                   cusparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const cusparseMatDescr_t descrA,
                                   cuDoubleComplex*         bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   cusparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 308
hipsparseStatus_t cusparseCreateBsric02Info(bsric02Info_t* info) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDestroyBsric02Info(bsric02Info_t info) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
hipsparseStatus_t cusparseXbsric02_zeroPivot(cusparseHandle_t handle,
                                            bsric02Info_t    info,
                                            int*             position) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSbsric02_bufferSize(cusparseHandle_t         handle,
                                             cusparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const cusparseMatDescr_t descrA,
                                             float*                   bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDbsric02_bufferSize(cusparseHandle_t         handle,
                                             cusparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const cusparseMatDescr_t descrA,
                                             double*                  bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCbsric02_bufferSize(cusparseHandle_t         handle,
                                             cusparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const cusparseMatDescr_t descrA,
                                             cuComplex*               bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZbsric02_bufferSize(cusparseHandle_t         handle,
                                             cusparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const cusparseMatDescr_t descrA,
                                             cuDoubleComplex*         bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSbsric02_analysis(cusparseHandle_t         handle,
                                           cusparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const cusparseMatDescr_t descrA,
                                           const float*             bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDbsric02_analysis(cusparseHandle_t         handle,
                                           cusparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const cusparseMatDescr_t descrA,
                                           const double*            bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCbsric02_analysis(cusparseHandle_t         handle,
                                           cusparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const cusparseMatDescr_t descrA,
                                           const cuComplex*         bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZbsric02_analysis(cusparseHandle_t         handle,
                                           cusparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const cusparseMatDescr_t descrA,
                                           const cuDoubleComplex*   bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSbsric02(cusparseHandle_t         handle,
                                  cusparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const cusparseMatDescr_t descrA,
                                  float*                   bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDbsric02(cusparseHandle_t         handle,
                                  cusparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const cusparseMatDescr_t descrA,
                                  double*                  bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCbsric02(cusparseHandle_t         handle,
                                  cusparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const cusparseMatDescr_t descrA,
                                  cuComplex*               bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*
                                       bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZbsric02(cusparseHandle_t         handle,
                                  cusparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const cusparseMatDescr_t descrA,
                                  cuDoubleComplex*         bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 400
hipsparseStatus_t cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                float*           boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                double*          boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuComplex*       boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                cuDoubleComplex* boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif



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

#if HIP_VERSION < 402
hipsparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr,
                                     int64_t               size,
                                     int64_t               nnz,
                                     void*                 indices,
                                     void*                 values,
                                     cusparseIndexType_t   idxType,
                                     cusparseIndexBase_t   idxBase,
                                     cudaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr,
                                  int64_t*             size,
                                  int64_t*             nnz,
                                  void**               indices,
                                  void**               values,
                                  cusparseIndexType_t* idxType,
                                  cusparseIndexBase_t* idxBase,
                                  cudaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr,
                                           cusparseIndexBase_t* idxBase) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr,
                                        void**               values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr,
                                        void*                values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 cooRowInd,
                                   void*                 cooColInd,
                                   void*                 cooValues,
                                   cusparseIndexType_t   cooIdxType,
                                   cusparseIndexBase_t   idxBase,
                                   cudaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCreateCooAoS(cusparseSpMatDescr_t* spMatDescr,
                                      int64_t               rows,
                                      int64_t               cols,
                                      int64_t               nnz,
                                      void*                 cooInd,
                                      void*                 cooValues,
                                      cusparseIndexType_t   cooIdxType,
                                      cusparseIndexBase_t   idxBase,
                                      cudaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 csrRowOffsets,
                                   void*                 csrColInd,
                                   void*                 csrValues,
                                   cusparseIndexType_t   csrRowOffsetsType,
                                   cusparseIndexType_t   csrColIndType,
                                   cusparseIndexBase_t   idxBase,
                                   cudaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 cscColOffsets,
                                   void*                 cscRowInd,
                                   void*                 cscValues,
                                   cusparseIndexType_t   cscColOffsetsType,
                                   cusparseIndexType_t   cscRowIndType,
                                   cusparseIndexBase_t   idxBase,
                                   cudaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCooGet(cusparseSpMatDescr_t spMatDescr,
                                int64_t*             rows,
                                int64_t*             cols,
                                int64_t*             nnz,
                                void**               cooRowInd,  // COO row indices
                                void**               cooColInd,  // COO column indices
                                void**               cooValues,  // COO values
                                cusparseIndexType_t* idxType,
                                cusparseIndexBase_t* idxBase,
                                cudaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr,
                                   int64_t*             rows,
                                   int64_t*             cols,
                                   int64_t*             nnz,
                                   void**               cooInd,     // COO indices
                                   void**               cooValues,  // COO values
                                   cusparseIndexType_t* idxType,
                                   cusparseIndexBase_t* idxBase,
                                   cudaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCsrGet(cusparseSpMatDescr_t spMatDescr,
                                int64_t*             rows,
                                int64_t*             cols,
                                int64_t*             nnz,
                                void**               csrRowOffsets,
                                void**               csrColInd,
                                void**               csrValues,
                                cusparseIndexType_t* csrRowOffsetsType,
                                cusparseIndexType_t* csrColIndType,
                                cusparseIndexBase_t* idxBase,
                                cudaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr,
                                        void*                csrRowOffsets,
                                        void*                csrColInd,
                                        void*                csrValues) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr,
                                      int64_t*             rows,
                                      int64_t*             cols,
                                      int64_t*             nnz) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMatGetFormat(cusparseSpMatDescr_t spMatDescr,
                                        cusparseFormat_t*    format) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMatGetIndexBase(cusparseSpMatDescr_t spMatDescr,
                                           cusparseIndexBase_t* idxBase) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr,
                                        void**               values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr,
                                        void*                values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                                     int64_t               size,
                                     void*                 values,
                                     cudaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr,
                                  int64_t*             size,
                                  void**               values,
                                  cudaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr,
                                        void**               values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr,
                                        void*                values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
                                     int64_t               rows,
                                     int64_t               cols,
                                     int64_t               ld,
                                     void*                 values,
                                     cudaDataType          valueType,
                                     cusparseOrder_t       order) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr,
                                  int64_t*             rows,
                                  int64_t*             cols,
                                  int64_t*             ld,
                                  void**               values,
                                  cudaDataType*        type,
                                  cusparseOrder_t*     order) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr,
                                        void**               values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr,
                                        void*                values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpVV_bufferSize(cusparseHandle_t     handle,
                                         cusparseOperation_t  opX,
                                         cusparseSpVecDescr_t vecX,
                                         cusparseDnVecDescr_t vecY,
                                         const void*          result,
                                         cudaDataType         computeType,
                                         size_t*              bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpVV(cusparseHandle_t     handle,
                              cusparseOperation_t  opX,
                              cusparseSpVecDescr_t vecX,
                              cusparseDnVecDescr_t vecY,
                              void*                result,
                              cudaDataType         computeType,
                              void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t    handle,
                                         cusparseOperation_t opA,
                                         const void*         alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseDnVecDescr_t vecX,
                                         const void*          beta,
                                         cusparseDnVecDescr_t vecY,
                                         cudaDataType         computeType,
                                         cusparseSpMVAlg_t    alg,
                                         size_t*              bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMV(cusparseHandle_t     handle,
                              cusparseOperation_t  opA,
                              const void*          alpha,
                              cusparseSpMatDescr_t matA,
                              cusparseDnVecDescr_t vecX,
                              const void*          beta,
                              cusparseDnVecDescr_t vecY,
                              cudaDataType         computeType,
                              cusparseSpMVAlg_t    alg,
                              void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t     handle,
                                         cusparseOperation_t  opA,
                                         cusparseOperation_t  opB,
                                         const void*          alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseDnMatDescr_t matB,
                                         const void*          beta,
                                         cusparseDnMatDescr_t matC,
                                         cudaDataType         computeType,
                                         cusparseSpMMAlg_t    alg,
                                         size_t*              bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpMM(cusparseHandle_t     handle,
                              cusparseOperation_t  opA,
                              cusparseOperation_t  opB,
                              const void*          alpha,
                              cusparseSpMatDescr_t matA,
                              cusparseDnMatDescr_t matB,
                              const void*          beta,
                              cusparseDnMatDescr_t matC,
                              cudaDataType         computeType,
                              cusparseSpMMAlg_t    alg,
                              void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t           handle,
                                                  cusparseSpMatDescr_t       matA,
                                                  cusparseDnMatDescr_t       matB,
                                                  cusparseSparseToDenseAlg_t alg,
                                                  size_t*                    bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSparseToDense(cusparseHandle_t           handle,
                                       cusparseSpMatDescr_t       matA,
                                       cusparseDnMatDescr_t       matB,
                                       cusparseSparseToDenseAlg_t alg,
                                       void*                      buffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t           handle,
                                                  cusparseDnMatDescr_t       matA,
                                                  cusparseSpMatDescr_t       matB,
                                                  cusparseDenseToSparseAlg_t alg,
                                                  size_t*                    bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t           handle,
                                                cusparseDnMatDescr_t       matA,
                                                cusparseSpMatDescr_t       matB,
                                                cusparseDenseToSparseAlg_t alg,
                                                void*                      buffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t           handle,
                                               cusparseDnMatDescr_t       matA,
                                               cusparseSpMatDescr_t       matB,
                                               cusparseDenseToSparseAlg_t alg,
                                               void*                      buffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 403
hipsparseStatus_t cusparseSgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 405
hipsparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 50000000
hipsparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t     spMatDescr,
                                           cusparseSpMatAttribute_t attribute,
                                           void*                    data,
                                           size_t                   dataSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t     handle,
                                         cusparseOperation_t  opA,
                                         cusparseOperation_t  opB,
                                         const void*          alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseDnMatDescr_t matB,
                                         cusparseDnMatDescr_t matC,
                                         cudaDataType         computeType,
                                         cusparseSpSMAlg_t    alg,
                                         cusparseSpSMDescr_t  spsmDescr,
                                         size_t*              bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSpSM_analysis(cusparseHandle_t     handle,
                                       cusparseOperation_t  opA,
                                       cusparseOperation_t  opB,
                                       const void*          alpha,
                                       cusparseSpMatDescr_t matA,
                                       cusparseDnMatDescr_t matB,
                                       cusparseDnMatDescr_t matC,
                                       cudaDataType         computeType,
                                       cusparseSpSMAlg_t    alg,
                                       cusparseSpSMDescr_t  spsmDescr,
                                       void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

// See cusparse.pyx for a comment
hipsparseStatus_t cusparseSpSM_solve(cusparseHandle_t     handle,
                                    cusparseOperation_t  opA,
                                    cusparseOperation_t  opB,
                                    const void*          alpha,
                                    cusparseSpMatDescr_t matA,
                                    cusparseDnMatDescr_t matB,
                                    cusparseDnMatDescr_t matC,
                                    cudaDataType         computeType,
                                    cusparseSpSMAlg_t    alg,
                                    cusparseSpSMDescr_t  spsmDescr,
                                    void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 501
hipsparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgtsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseSgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseDgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseCgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t cusparseZgpsvInterleavedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

#endif
}  // extern "C"

#endif  // INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
