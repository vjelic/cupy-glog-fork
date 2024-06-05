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

#if HIP_VERSION < 40100000
#define HIPSPARSE_STATUS_NOT_SUPPORTED (hipsparseStatus_t)10
#endif

extern "C" {
#if HIP_VERSION < 30800000
typedef void* bsric02Info_t;
#endif

#if HIP_VERSION < 30900000
typedef void* bsrilu02Info_t;
#endif
typedef enum {} cusparseAlgMode_t;

#if HIP_VERSION < 60000000
// Error handling
const char* hipsparseGetErrorName(...) {
    // Unavailable in hipSparse; this should not be called
    return "CUPY_HIPSPARSE_BINDING_UNEXPECTED_ERROR";
}
const char* hipsparseGetErrorString(...) {
    // Unavailable in hipSparse; this should not be called
    return "unexpected error in CuPy hipSparse binding";
}
#endif

#if HIP_VERSION < 54000000
typedef enum {} hipsparseCsr2CscAlg_t;
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
#if HIP_VERSION < 30900000
hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
hipsparseStatus_t hipsparseSbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                float*           boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                double*          boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                hipComplex*       boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                bsrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                hipDoubleComplex* boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseXbsrilu02_zeroPivot(hipsparseHandle_t handle,
                                             bsrilu02Info_t   info,
                                             int*             position) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                   bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                  bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                      mb,
                                              int                      nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedVal,
                                              const int*               bsrSortedRowPtr,
                                              const int*               bsrSortedColInd,
                                              int                      blockDim,
                                              bsrilu02Info_t           info,
                                              int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSbsrilu02_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            float*                   bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDbsrilu02_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            double*                  bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCbsrilu02_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            hipComplex*               bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZbsrilu02_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            int                      mb,
                                            int                      nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            hipDoubleComplex*         bsrSortedVal,
                                            const int*               bsrSortedRowPtr,
                                            const int*               bsrSortedColInd,
                                            int                      blockDim,
                                            bsrilu02Info_t           info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSbsrilu02(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const hipsparseMatDescr_t descrA,
                                   float*                   bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   hipsparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDbsrilu02(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const hipsparseMatDescr_t descrA,
                                   double*                  bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   hipsparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCbsrilu02(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const hipsparseMatDescr_t descrA,
                                   hipComplex*               bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   hipsparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZbsrilu02(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dirA,
                                   int                      mb,
                                   int                      nnzb,
                                   const hipsparseMatDescr_t descrA,
                                   hipDoubleComplex*         bsrSortedVal,
                                   const int*               bsrSortedRowPtr,
                                   const int*               bsrSortedColInd,
                                   int                      blockDim,
                                   bsrilu02Info_t           info,
                                   hipsparseSolvePolicy_t    policy,
                                   void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 30800000
hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
hipsparseStatus_t hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle,
                                            bsric02Info_t    info,
                                            int*             position) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSbsric02_bufferSize(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             float*                   bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDbsric02_bufferSize(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             double*                  bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCbsric02_bufferSize(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             hipComplex*               bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZbsric02_bufferSize(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                      mb,
                                             int                      nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             hipDoubleComplex*         bsrSortedVal,
                                             const int*               bsrSortedRowPtr,
                                             const int*               bsrSortedColInd,
                                             int                      blockDim,
                                             bsric02Info_t            info,
                                             int*                     pBufferSizeInBytes) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSbsric02_analysis(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const float*             bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           hipsparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDbsric02_analysis(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const double*            bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           hipsparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCbsric02_analysis(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const hipComplex*         bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           hipsparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZbsric02_analysis(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                      mb,
                                           int                      nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const hipDoubleComplex*   bsrSortedVal,
                                           const int*               bsrSortedRowPtr,
                                           const int*               bsrSortedColInd,
                                           int                      blockDim,
                                           bsric02Info_t            info,
                                           hipsparseSolvePolicy_t    policy,
                                           void*                    pInputBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSbsric02(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const hipsparseMatDescr_t descrA,
                                  float*                   bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  hipsparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDbsric02(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const hipsparseMatDescr_t descrA,
                                  double*                  bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  hipsparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCbsric02(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const hipsparseMatDescr_t descrA,
                                  hipComplex*               bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*
                                       bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  hipsparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZbsric02(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  int                      mb,
                                  int                      nnzb,
                                  const hipsparseMatDescr_t descrA,
                                  hipDoubleComplex*         bsrSortedVal,
                                  const int*               bsrSortedRowPtr,
                                  const int*               bsrSortedColInd,
                                  int                      blockDim,
                                  bsric02Info_t            info,
                                  hipsparseSolvePolicy_t    policy,
                                  void*                    pBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 40000000
hipsparseStatus_t hipsparseScsrilu02_numericBoost(hipsparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                float*           boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                double*          boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                hipComplex*       boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                csrilu02Info_t   info,
                                                int              enable_boost,
                                                double*          tol,
                                                hipDoubleComplex* boost_val) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif



#define CUSPARSE_VERSION (hipsparseVersionMajor*100000+hipsparseVersionMinor*100+hipsparseVersionPatch)

// hipSPARSE generic API
#if HIP_VERSION < 40200000
typedef void* hipsparseSpVecDescr_t;
typedef void* hipsparseDnVecDescr_t;
typedef void* hipsparseSpMatDescr_t;
typedef void* hipsparseDnMatDescr_t;
typedef enum {} hipsparseIndexType_t;
typedef enum {} hipsparseFormat_t;
typedef enum {} hipsparseOrder_t;
typedef enum {} hipsparseSpMVAlg_t;
typedef enum {} hipsparseSpMMAlg_t;
typedef enum {} hipsparseSparseToDenseAlg_t;
typedef enum {} hipsparseDenseToSparseAlg_t;
#endif

#if HIP_VERSION < 50000000
typedef enum {} hipsparseSpMatAttribute_t;
typedef enum {} hipsparseSpSMAlg_t;
typedef void * hipsparseSpSMDescr_t;
#endif

#if HIP_VERSION < 40200000
hipsparseStatus_t hipsparseCreateSpVec(hipsparseSpVecDescr_t* spVecDescr,
                                     int64_t               size,
                                     int64_t               nnz,
                                     void*                 indices,
                                     void*                 values,
                                     hipsparseIndexType_t   idxType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipdaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDestroySpVec(hipsparseSpVecDescr_t spVecDescr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpVecGet(hipsparseSpVecDescr_t spVecDescr,
                                  int64_t*             size,
                                  int64_t*             nnz,
                                  void**               indices,
                                  void**               values,
                                  hipsparseIndexType_t* idxType,
                                  hipsparseIndexBase_t* idxBase,
                                  hipdaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpVecGetIndexBase(hipsparseSpVecDescr_t spVecDescr,
                                           hipsparseIndexBase_t* idxBase) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpVecGetValues(hipsparseSpVecDescr_t spVecDescr,
                                        void**               values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpVecSetValues(hipsparseSpVecDescr_t spVecDescr,
                                        void*                values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCreateCoo(hipsparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 cooRowInd,
                                   void*                 cooColInd,
                                   void*                 cooValues,
                                   hipsparseIndexType_t   cooIdxType,
                                   hipsparseIndexBase_t   idxBase,
                                   hipdaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCreateCooAoS(hipsparseSpMatDescr_t* spMatDescr,
                                      int64_t               rows,
                                      int64_t               cols,
                                      int64_t               nnz,
                                      void*                 cooInd,
                                      void*                 cooValues,
                                      hipsparseIndexType_t   cooIdxType,
                                      hipsparseIndexBase_t   idxBase,
                                      hipdaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCreateCsr(hipsparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 csrRowOffsets,
                                   void*                 csrColInd,
                                   void*                 csrValues,
                                   hipsparseIndexType_t   csrRowOffsetsType,
                                   hipsparseIndexType_t   csrColIndType,
                                   hipsparseIndexBase_t   idxBase,
                                   hipdaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCreateCsc(hipsparseSpMatDescr_t* spMatDescr,
                                   int64_t               rows,
                                   int64_t               cols,
                                   int64_t               nnz,
                                   void*                 cscColOffsets,
                                   void*                 cscRowInd,
                                   void*                 cscValues,
                                   hipsparseIndexType_t   cscColOffsetsType,
                                   hipsparseIndexType_t   cscRowIndType,
                                   hipsparseIndexBase_t   idxBase,
                                   hipdaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDestroySpMat(hipsparseSpMatDescr_t spMatDescr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCooGet(hipsparseSpMatDescr_t spMatDescr,
                                int64_t*             rows,
                                int64_t*             cols,
                                int64_t*             nnz,
                                void**               cooRowInd,  // COO row indices
                                void**               cooColInd,  // COO column indices
                                void**               cooValues,  // COO values
                                hipsparseIndexType_t* idxType,
                                hipsparseIndexBase_t* idxBase,
                                hipdaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCooAoSGet(hipsparseSpMatDescr_t spMatDescr,
                                   int64_t*             rows,
                                   int64_t*             cols,
                                   int64_t*             nnz,
                                   void**               cooInd,     // COO indices
                                   void**               cooValues,  // COO values
                                   hipsparseIndexType_t* idxType,
                                   hipsparseIndexBase_t* idxBase,
                                   hipdaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCsrGet(hipsparseSpMatDescr_t spMatDescr,
                                int64_t*             rows,
                                int64_t*             cols,
                                int64_t*             nnz,
                                void**               csrRowOffsets,
                                void**               csrColInd,
                                void**               csrValues,
                                hipsparseIndexType_t* csrRowOffsetsType,
                                hipsparseIndexType_t* csrColIndType,
                                hipsparseIndexBase_t* idxBase,
                                hipdaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                        void*                csrRowOffsets,
                                        void*                csrColInd,
                                        void*                csrValues) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMatGetSize(hipsparseSpMatDescr_t spMatDescr,
                                      int64_t*             rows,
                                      int64_t*             cols,
                                      int64_t*             nnz) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMatGetFormat(hipsparseSpMatDescr_t spMatDescr,
                                        hipsparseFormat_t*    format) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMatGetIndexBase(hipsparseSpMatDescr_t spMatDescr,
                                           hipsparseIndexBase_t* idxBase) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMatGetValues(hipsparseSpMatDescr_t spMatDescr,
                                        void**               values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMatSetValues(hipsparseSpMatDescr_t spMatDescr,
                                        void*                values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr,
                                     int64_t               size,
                                     void*                 values,
                                     hipdaDataType          valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDnVecGet(hipsparseDnVecDescr_t dnVecDescr,
                                  int64_t*             size,
                                  void**               values,
                                  hipdaDataType*        valueType) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDnVecGetValues(hipsparseDnVecDescr_t dnVecDescr,
                                        void**               values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDnVecSetValues(hipsparseDnVecDescr_t dnVecDescr,
                                        void*                values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr,
                                     int64_t               rows,
                                     int64_t               cols,
                                     int64_t               ld,
                                     void*                 values,
                                     hipdaDataType          valueType,
                                     hipsparseOrder_t       order) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDnMatGet(hipsparseDnMatDescr_t dnMatDescr,
                                  int64_t*             rows,
                                  int64_t*             cols,
                                  int64_t*             ld,
                                  void**               values,
                                  hipdaDataType*        type,
                                  hipsparseOrder_t*     order) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDnMatGetValues(hipsparseDnMatDescr_t dnMatDescr,
                                        void**               values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDnMatSetValues(hipsparseDnMatDescr_t dnMatDescr,
                                        void*                values) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                         hipsparseOperation_t  opX,
                                         hipsparseSpVecDescr_t vecX,
                                         hipsparseDnVecDescr_t vecY,
                                         const void*          result,
                                         hipdaDataType         computeType,
                                         size_t*              bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                              hipsparseOperation_t  opX,
                              hipsparseSpVecDescr_t vecX,
                              hipsparseDnVecDescr_t vecY,
                              void*                result,
                              hipdaDataType         computeType,
                              void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t    handle,
                                         hipsparseOperation_t opA,
                                         const void*         alpha,
                                         hipsparseSpMatDescr_t matA,
                                         hipsparseDnVecDescr_t vecX,
                                         const void*          beta,
                                         hipsparseDnVecDescr_t vecY,
                                         hipdaDataType         computeType,
                                         hipsparseSpMVAlg_t    alg,
                                         size_t*              bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t     handle,
                              hipsparseOperation_t  opA,
                              const void*          alpha,
                              hipsparseSpMatDescr_t matA,
                              hipsparseDnVecDescr_t vecX,
                              const void*          beta,
                              hipsparseDnVecDescr_t vecY,
                              hipdaDataType         computeType,
                              hipsparseSpMVAlg_t    alg,
                              void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t     handle,
                                         hipsparseOperation_t  opA,
                                         hipsparseOperation_t  opB,
                                         const void*          alpha,
                                         hipsparseSpMatDescr_t matA,
                                         hipsparseDnMatDescr_t matB,
                                         const void*          beta,
                                         hipsparseDnMatDescr_t matC,
                                         hipdaDataType         computeType,
                                         hipsparseSpMMAlg_t    alg,
                                         size_t*              bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t     handle,
                              hipsparseOperation_t  opA,
                              hipsparseOperation_t  opB,
                              const void*          alpha,
                              hipsparseSpMatDescr_t matA,
                              hipsparseDnMatDescr_t matB,
                              const void*          beta,
                              hipsparseDnMatDescr_t matC,
                              hipdaDataType         computeType,
                              hipsparseSpMMAlg_t    alg,
                              void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t           handle,
                                                  hipsparseSpMatDescr_t       matA,
                                                  hipsparseDnMatDescr_t       matB,
                                                  hipsparseSparseToDenseAlg_t alg,
                                                  size_t*                    bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t           handle,
                                       hipsparseSpMatDescr_t       matA,
                                       hipsparseDnMatDescr_t       matB,
                                       hipsparseSparseToDenseAlg_t alg,
                                       void*                      buffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                  hipsparseDnMatDescr_t       matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  size_t*                    bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                hipsparseDnMatDescr_t       matA,
                                                hipsparseSpMatDescr_t       matB,
                                                hipsparseDenseToSparseAlg_t alg,
                                                void*                      buffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                               hipsparseDnMatDescr_t       matA,
                                               hipsparseSpMatDescr_t       matB,
                                               hipsparseDenseToSparseAlg_t alg,
                                               void*                      buffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 40300000
hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgtsv2(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgtsv2_nopivot(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 40500000
hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseDgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseCgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseZgtsv2StridedBatch(...) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 50000000
hipsparseStatus_t hipsparseSpMatSetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                           hipsparseSpMatAttribute_t attribute,
                                           void*                    data,
                                           size_t                   dataSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t     handle,
                                         hipsparseOperation_t  opA,
                                         hipsparseOperation_t  opB,
                                         const void*          alpha,
                                         hipsparseSpMatDescr_t matA,
                                         hipsparseDnMatDescr_t matB,
                                         hipsparseDnMatDescr_t matC,
                                         hipdaDataType         computeType,
                                         hipsparseSpSMAlg_t    alg,
                                         hipsparseSpSMDescr_t  spsmDescr,
                                         size_t*              bufferSize) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t     handle,
                                       hipsparseOperation_t  opA,
                                       hipsparseOperation_t  opB,
                                       const void*          alpha,
                                       hipsparseSpMatDescr_t matA,
                                       hipsparseDnMatDescr_t matB,
                                       hipsparseDnMatDescr_t matC,
                                       hipdaDataType         computeType,
                                       hipsparseSpSMAlg_t    alg,
                                       hipsparseSpSMDescr_t  spsmDescr,
                                       void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}

// See hipsparse.pyx for a comment
hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t     handle,
                                    hipsparseOperation_t  opA,
                                    hipsparseOperation_t  opB,
                                    const void*          alpha,
                                    hipsparseSpMatDescr_t matA,
                                    hipsparseDnMatDescr_t matB,
                                    hipsparseDnMatDescr_t matC,
                                    hipdaDataType         computeType,
                                    hipsparseSpSMAlg_t    alg,
                                    hipsparseSpSMDescr_t  spsmDescr,
                                    void*                externalBuffer) {
  return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if HIP_VERSION < 50100000
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

#endif
}  // extern "C"

#endif  // INCLUDE_GUARD_HIP_CUPY_HIPSPARSE_H
