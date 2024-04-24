#ifndef INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H
#define INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H

#include "cupy_hip.h"
#include "cupy_hipblas.h"
#include <stdexcept>  // for gcc 10.0

extern "C" {

hipsolverStatus_t cusolverGetProperty(hipLibraryPropertyType_t type, int* val) {
    switch(type) {
        case MAJOR_VERSION: { *val = hipsolverVersionMajor; break; }
        case MINOR_VERSION: { *val = hipsolverVersionMinor; break; }
        case PATCH_LEVEL:   { *val = hipsolverVersionPatch; break; }
        default: throw std::runtime_error("invalid type");
    }
    return HIPSOLVER_STATUS_SUCCESS;
}

#if HIP_VERSION < 60240092
typedef enum hipsolverDnParams_t {};
#endif

#if HIP_VERSION < 50631061
typedef hipsolverHandle_t hipsolverDnHandle_t;
typedef void* hipsolverGesvdjInfo_t;
typedef void* hipsolverSyevjInfo_t;

hipsolverStatus_t hipsolverDnSorgqr(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    int               k,
                                    float*            A,
                                    int               lda,
                                    const float*      tau,
                                    float*            work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDorgqr(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    int               k,
                                    double*           A,
                                    int               lda,
                                    const double*     tau,
                                    double*           work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCungqr(hipsolverHandle_t      handle,
                                    int                    m,
                                    int                    n,
                                    int                    k,
                                    hipFloatComplex*       A,
                                    int                    lda,
                                    const hipFloatComplex* tau,
                                    hipFloatComplex*       work,
                                    int                    lwork,
                                    int*                   devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZungqr(hipsolverHandle_t       handle,
                                    int                     m,
                                    int                     n,
                                    int                     k,
                                    hipDoubleComplex*       A,
                                    int                     lda,
                                    const hipDoubleComplex* tau,
                                    hipDoubleComplex*       work,
                                    int                     lwork,
                                    int*                    devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDestroy(hipsolverHandle_t handle) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCreate(hipsolverHandle_t* handle) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSetStream(hipsolverHandle_t handle,
                                       hipStream_t       streamId) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnGetStream(hipsolverHandle_t handle,
                                       hipStream_t*      streamId) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCpotrf_bufferSize(hipsolverHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int               n,
                                               hipFloatComplex* A,
                                               int               lda,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZpotrf_bufferSize(hipsolverHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int               n,
                                               hipDoubleComplex* A,
                                               int               lda,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSpotrf(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    float*            A,
                                    int               lda,
                                    float*            work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDpotrf(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    double*           A,
                                    int               lda,
                                    double*           work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCpotrf(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    hipFloatComplex*  A,
                                    int               lda,
                                    hipFloatComplex*  work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZpotrf(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    hipDoubleComplex* A,
                                    int               lda,
                                    hipDoubleComplex* work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSpotrs(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    int               nrhs,
                                    const float*      A,
                                    int               lda,
                                    float*            B,
                                    int               ldb,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDpotrs(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    int               nrhs,
                                    const double*     A,
                                    int               lda,
                                    double*           B,
                                    int               ldb,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCpotrs(hipsolverHandle_t      handle,
                                    hipsolverFillMode_t      uplo,
                                    int                    n,
                                    int                    nrhs,
                                    const hipFloatComplex* A,
                                    int                    lda,
                                    hipFloatComplex*       B,
                                    int                    ldb,
                                    int*                   devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZpotrs(hipsolverHandle_t       handle,
                                    hipsolverFillMode_t       uplo,
                                    int                     n,
                                    int                     nrhs,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    hipDoubleComplex*       B,
                                    int                     ldb,
                                    int*                    devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSpotrfBatched(hipsolverHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int               n,
                                           float*            A[],
                                           int               lda,
                                           int*              devInfo,
                                           int               batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDpotrfBatched(hipsolverHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int               n,
                                           double*           A[],
                                           int               lda,
                                           int*              devInfo,
                                           int               batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCpotrfBatched(hipsolverHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int               n,
                                           hipFloatComplex*  A[],
                                           int               lda,
                                           int*              devInfo,
                                           int               batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZpotrfBatched(hipsolverHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int               n,
                                           hipDoubleComplex* A[],
                                           int               lda,
                                           int*              devInfo,
                                           int               batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSpotrsBatched(hipsolverHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int               n,
                                           int               nrhs,
                                           float*            A[],
                                           int               lda,
                                           float*            B[],
                                           int               ldb,
                                           int*              devInfo,
                                           int               batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDpotrsBatched(hipsolverHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int               n,
                                           int               nrhs,
                                           double*           A[],
                                           int               lda,
                                           double*           B[],
                                           int               ldb,
                                           int*              devInfo,
                                           int               batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCpotrsBatched(hipsolverHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int               n,
                                           int               nrhs,
                                           hipFloatComplex*  A[],
                                           int               lda,
                                           hipFloatComplex*  B[],
                                           int               ldb,
                                           int*              devInfo,
                                           int               batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZpotrsBatched(hipsolverHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int               n,
                                           int               nrhs,
                                           hipDoubleComplex* A[],
                                           int               lda,
                                           hipDoubleComplex* B[],
                                           int               ldb,
                                           int*              devInfo,
                                           int               batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgetrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    float*            A,
                                    int               lda,
                                    float*            work,
                                    int*              devIpiv,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgetrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    double*           A,
                                    int               lda,
                                    double*           work,
                                    int*              devIpiv,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgetrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    hipFloatComplex*  A,
                                    int               lda,
                                    hipFloatComplex*  work,
                                    int*              devIpiv,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgetrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    hipDoubleComplex* A,
                                    int               lda,
                                    hipDoubleComplex* work,
                                    int*              devIpiv,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgetrs(hipsolverHandle_t  handle,
                                    hipsolverOperation_t trans,
                                    int                n,
                                    int                nrhs,
                                    const float*       A,
                                    int                lda,
                                    const int*         devIpiv,
                                    float*             B,
                                    int                ldb,
                                    int*               devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgetrs(hipsolverHandle_t  handle,
                                    hipsolverOperation_t trans,
                                    int                n,
                                    int                nrhs,
                                    const double*      A,
                                    int                lda,
                                    const int*         devIpiv,
                                    double*            B,
                                    int                ldb,
                                    int*               devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgetrs(hipsolverHandle_t      handle,
                                     hipsolverOperation_t     trans,
                                     int                    n,
                                     int                    nrhs,
                                     const hipFloatComplex* A,
                                     int                    lda,
                                     const int*             devIpiv,
                                     hipFloatComplex*       B,
                                     int                    ldb,
                                     int*                   devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgetrs(hipsolverHandle_t       handle,
                                    hipsolverOperation_t      trans,
                                    int                     n,
                                    int                     nrhs,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const int*              devIpiv,
                                    hipDoubleComplex*       B,
                                    int                     ldb,
                                    int*                    devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgeqrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    float*            A,
                                    int               lda,
                                    float*            tau,
                                    float*            work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgeqrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    double*           A,
                                    int               lda,
                                    double*           tau,
                                    double*           work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgeqrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    hipFloatComplex*  A,
                                    int               lda,
                                    hipFloatComplex*  tau,
                                    hipFloatComplex*  work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgeqrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    hipDoubleComplex* A,
                                    int               lda,
                                    hipDoubleComplex* tau,
                                    hipDoubleComplex* work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSorgqr_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               const float*      A,
                                               int               lda,
                                               const float*      tau,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDorgqr_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               const double*     A,
                                               int               lda,
                                               const double*     tau,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCungqr_bufferSize(hipsolverHandle_t      handle,
                                               int                    m,
                                               int                    n,
                                               int                    k,
                                               const hipFloatComplex* A,
                                               int                    lda,
                                               const hipFloatComplex* tau,
                                               int*                   lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZungqr_bufferSize(hipsolverHandle_t       handle,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               const hipDoubleComplex* tau,
                                               int*                    lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSormqr_bufferSize(hipsolverHandle_t  handle,
                                               hipsolverSideMode_t  side,
                                               hipsolverOperation_t trans,
                                               int                m,
                                               int                n,
                                               int                k,
                                               const float*       A,
                                               int                lda,
                                               const float*       tau,
                                               const float*       C,
                                               int                ldc,
                                               int*               lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDormqr_bufferSize(hipsolverHandle_t  handle,
                                               hipsolverSideMode_t  side,
                                               hipsolverOperation_t trans,
                                               int                m,
                                               int                n,
                                               int                k,
                                               const double*      A,
                                               int                lda,
                                               const double*      tau,
                                               const double*      C,
                                               int                ldc,
                                               int*               lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCunmqr_bufferSize(hipsolverHandle_t      handle,
                                               hipsolverSideMode_t      side,
                                               hipsolverOperation_t     trans,
                                               int                    m,
                                               int                    n,
                                               int                    k,
                                               const hipFloatComplex* A,
                                               int                    lda,
                                               const hipFloatComplex* tau,
                                               const hipFloatComplex* C,
                                               int                    ldc,
                                               int*                   lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZunmqr_bufferSize(hipsolverHandle_t       handle,
                                               hipsolverSideMode_t       side,
                                               hipsolverOperation_t      trans,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               const hipDoubleComplex* tau,
                                               const hipDoubleComplex* C,
                                               int                     ldc,
                                               int*                    lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSormqr(hipsolverHandle_t  handle,
                                    hipsolverSideMode_t  side,
                                    hipsolverOperation_t trans,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const float*       A,
                                    int                lda,
                                    const float*       tau,
                                    float*             C,
                                    int                ldc,
                                    float*             work,
                                    int                lwork,
                                    int*               devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDormqr(hipsolverHandle_t  handle,
                                    hipsolverSideMode_t  side,
                                    hipsolverOperation_t trans,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const double*      A,
                                    int                lda,
                                    const double*      tau,
                                    double*            C,
                                    int                ldc,
                                    double*            work,
                                    int                lwork,
                                    int*               devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCunmqr(hipsolverHandle_t      handle,
                                    hipsolverSideMode_t      side,
                                    hipsolverOperation_t     trans,
                                    int                    m,
                                    int                    n,
                                    int                    k,
                                    const hipFloatComplex* A,
                                    int                    lda,
                                    const hipFloatComplex* tau,
                                    hipFloatComplex*       C,
                                    int                    ldc,
                                    hipFloatComplex*       work,
                                    int                    lwork,
                                    int*                   devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZunmqr(hipsolverHandle_t       handle,
                                    hipsolverSideMode_t       side,
                                    hipsolverOperation_t      trans,
                                    int                     m,
                                    int                     n,
                                    int                     k,
                                    const hipDoubleComplex* A,
                                    int                     lda,
                                    const hipDoubleComplex* tau,
                                    hipDoubleComplex*       C,
                                    int                     ldc,
                                    hipDoubleComplex*       work,
                                    int                     lwork,
                                    int*                    devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSsytrf_bufferSize(hipsolverHandle_t handle, int n,
                                               float* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnDsytrf_bufferSize(hipsolverHandle_t handle, int n, double* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSsytrf(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    float*            A,
                                    int               lda,
                                    int*              ipiv,
                                    float*            work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDsytrf(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    double*           A,
                                    int               lda,
                                    int*              ipiv,
                                    double*           work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCsytrf(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    hipFloatComplex*  A,
                                    int               lda,
                                    int*              ipiv,
                                    hipFloatComplex*  work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZsytrf(hipsolverHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int               n,
                                    hipDoubleComplex* A,
                                    int               lda,
                                    int*              ipiv,
                                    hipDoubleComplex* work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgebrd_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgebrd_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgebrd_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgebrd_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgebrd(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    float*            A,
                                    int               lda,
                                    float*            D,
                                    float*            E,
                                    float*            tauq,
                                    float*            taup,
                                    float*            work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgebrd(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    double*           A,
                                    int               lda,
                                    double*           D,
                                    double*           E,
                                    double*           tauq,
                                    double*           taup,
                                    double*           work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgebrd(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    hipFloatComplex*  A,
                                    int               lda,
                                    float*            D,
                                    float*            E,
                                    hipFloatComplex*  tauq,
                                    hipFloatComplex*  taup,
                                    hipFloatComplex*  work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgebrd(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    hipDoubleComplex* A,
                                    int               lda,
                                    double*           D,
                                    double*           E,
                                    hipDoubleComplex* tauq,
                                    hipDoubleComplex* taup,
                                    hipDoubleComplex* work,
                                    int               lwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int*              lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgesvd(hipsolverHandle_t handle,
                                    signed char       jobu,
                                    signed char       jobv,
                                    int               m,
                                    int               n,
                                    float*            A,
                                    int               lda,
                                    float*            S,
                                    float*            U,
                                    int               ldu,
                                    float*            V,
                                    int               ldv,
                                    float*            work,
                                    int               lwork,
                                    float*            rwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgesvd(hipsolverHandle_t handle,
                                    signed char       jobu,
                                    signed char       jobv,
                                    int               m,
                                    int               n,
                                    double*           A,
                                    int               lda,
                                    double*           S,
                                    double*           U,
                                    int               ldu,
                                    double*           V,
                                    int               ldv,
                                    double*           work,
                                    int               lwork,
                                    double*           rwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgesvd(hipsolverHandle_t handle,
                                    signed char       jobu,
                                    signed char       jobv,
                                    int               m,
                                    int               n,
                                    hipFloatComplex*  A,
                                    int               lda,
                                    float*            S,
                                    hipFloatComplex*  U,
                                    int               ldu,
                                    hipFloatComplex*  V,
                                    int               ldv,
                                    hipFloatComplex*  work,
                                    int               lwork,
                                    float*            rwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgesvd(hipsolverHandle_t handle,
                                    signed char       jobu,
                                    signed char       jobv,
                                    int               m,
                                    int               n,
                                    hipDoubleComplex* A,
                                    int               lda,
                                    double*           S,
                                    hipDoubleComplex* U,
                                    int               ldu,
                                    hipDoubleComplex* V,
                                    int               ldv,
                                    hipDoubleComplex* work,
                                    int               lwork,
                                    double*           rwork,
                                    int*              devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCreateGesvdjInfo(hipsolverGesvdjInfo_t* info) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDestroyGesvdjInfo(hipsolverGesvdjInfo_t info) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXgesvdjSetTolerance(hipsolverGesvdjInfo_t info,
                                                 double                tolerance) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXgesvdjSetMaxSweeps(hipsolverGesvdjInfo_t info,
                                                 int                   max_sweeps) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXgesvdjSetSortEig(hipsolverGesvdjInfo_t info,
                                               int                   sort_eig) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXgesvdjGetResidual(hipsolverDnHandle_t   handle,
                                                hipsolverGesvdjInfo_t info,
                                                double*               residual) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXgesvdjGetSweeps(hipsolverDnHandle_t   handle,
                                              hipsolverGesvdjInfo_t info,
                                              int* executed_sweeps) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgesvdj_bufferSize(hipsolverDnHandle_t   handle,
                                                hipsolverEigMode_t    jobz,
                                                int                   econ,
                                                int                   m,
                                                int                   n,
                                                const float*          A,
                                                int                   lda,
                                                const float*          S,
                                                const float*          U,
                                                int                   ldu,
                                                const float*          V,
                                                int                   ldv,
                                                int*                  lwork,
                                                hipsolverGesvdjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgesvdj_bufferSize(hipsolverDnHandle_t   handle,
                                                hipsolverEigMode_t    jobz,
                                                int                   econ,
                                                int                   m,
                                                int                   n,
                                                const double*         A,
                                                int                   lda,
                                                const double*         S,
                                                const double*         U,
                                                int                   ldu,
                                                const double*         V,
                                                int                   ldv,
                                                int*                  lwork,
                                                hipsolverGesvdjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgesvdj_bufferSize(hipsolverDnHandle_t    handle,
                                                hipsolverEigMode_t     jobz,
                                                int                    econ,
                                                int                    m,
                                                int                    n,
                                                const hipFloatComplex* A,
                                                int                    lda,
                                                const float*           S,
                                                const hipFloatComplex* U,
                                                int                    ldu,
                                                const hipFloatComplex* V,
                                                int                    ldv,
                                                int*                   lwork,
                                                hipsolverGesvdjInfo_t  params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgesvdj_bufferSize(hipsolverDnHandle_t     handle,
                                                hipsolverEigMode_t      jobz,
                                                int                     econ,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* A,
                                                int                     lda,
                                                const double*           S,
                                                const hipDoubleComplex* U,
                                                int                     ldu,
                                                const hipDoubleComplex* V,
                                                int                     ldv,
                                                int*                    lwork,
                                                hipsolverGesvdjInfo_t   params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgesvdj(hipsolverDnHandle_t   handle,
                                     hipsolverEigMode_t    jobz,
                                     int                   econ,
                                     int                   m,
                                     int                   n,
                                     float*                A,
                                     int                   lda,
                                     float*                S,
                                     float*                U,
                                     int                   ldu,
                                     float*                V,
                                     int                   ldv,
                                     float*                work,
                                     int                   lwork,
                                     int*                  devInfo,
                                     hipsolverGesvdjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgesvdj(hipsolverDnHandle_t   handle,
                                     hipsolverEigMode_t    jobz,
                                     int                   econ,
                                     int                   m,
                                     int                   n,
                                     double*               A,
                                     int                   lda,
                                     double*               S,
                                     double*               U,
                                     int                   ldu,
                                     double*               V,
                                     int                   ldv,
                                     double*               work,
                                     int                   lwork,
                                     int*                  devInfo,
                                     hipsolverGesvdjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgesvdj(hipsolverDnHandle_t   handle,
                                     hipsolverEigMode_t    jobz,
                                     int                   econ,
                                     int                   m,
                                     int                   n,
                                     hipFloatComplex*      A,
                                     int                   lda,
                                     float*                S,
                                     hipFloatComplex*      U,
                                     int                   ldu,
                                     hipFloatComplex*      V,
                                     int                   ldv,
                                     hipFloatComplex*      work,
                                     int                   lwork,
                                     int*                  devInfo,
                                     hipsolverGesvdjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgesvdj(hipsolverDnHandle_t   handle,
                                     hipsolverEigMode_t    jobz,
                                     int                   econ,
                                     int                   m,
                                     int                   n,
                                     hipDoubleComplex*     A,
                                     int                   lda,
                                     double*               S,
                                     hipDoubleComplex*     U,
                                     int                   ldu,
                                     hipDoubleComplex*     V,
                                     int                   ldv,
                                     hipDoubleComplex*     work,
                                     int                   lwork,
                                     int*                  devInfo,
                                     hipsolverGesvdjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnSgesvdjBatched_bufferSize(hipsolverDnHandle_t   handle,
                                         hipsolverEigMode_t    jobz,
                                         int                   m,
                                         int                   n,
                                         const float*          A,
                                         int                   lda,
                                         const float*          S,
                                         const float*          U,
                                         int                   ldu,
                                         const float*          V,
                                         int                   ldv,
                                         int*                  lwork,
                                         hipsolverGesvdjInfo_t params,
                                         int                   batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnDgesvdjBatched_bufferSize(hipsolverDnHandle_t   handle,
                                         hipsolverEigMode_t    jobz,
                                         int                   m,
                                         int                   n,
                                         const double*         A,
                                         int                   lda,
                                         const double*         S,
                                         const double*         U,
                                         int                   ldu,
                                         const double*         V,
                                         int                   ldv,
                                         int*                  lwork,
                                         hipsolverGesvdjInfo_t params,
                                         int                   batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnCgesvdjBatched_bufferSize(hipsolverDnHandle_t    handle,
                                         hipsolverEigMode_t     jobz,
                                         int                    m,
                                         int                    n,
                                         const hipFloatComplex* A,
                                         int                    lda,
                                         const float*           S,
                                         const hipFloatComplex* U,
                                         int                    ldu,
                                         const hipFloatComplex* V,
                                         int                    ldv,
                                         int*                   lwork,
                                         hipsolverGesvdjInfo_t  params,
                                         int                    batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnZgesvdjBatched_bufferSize(hipsolverDnHandle_t     handle,
                                         hipsolverEigMode_t      jobz,
                                         int                     m,
                                         int                     n,
                                         const hipDoubleComplex* A,
                                         int                     lda,
                                         const double*           S,
                                         const hipDoubleComplex* U,
                                         int                     ldu,
                                         const hipDoubleComplex* V,
                                         int                     ldv,
                                         int*                    lwork,
                                         hipsolverGesvdjInfo_t   params,
                                         int                     batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgesvdjBatched(hipsolverDnHandle_t   handle,
                                            hipsolverEigMode_t    jobz,
                                            int                   m,
                                            int                   n,
                                            float*                A,
                                            int                   lda,
                                            float*                S,
                                            float*                U,
                                            int                   ldu,
                                            float*                V,
                                            int                   ldv,
                                            float*                work,
                                            int                   lwork,
                                            int*                  devInfo,
                                            hipsolverGesvdjInfo_t params,
                                            int                   batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgesvdjBatched(hipsolverDnHandle_t   handle,
                                            hipsolverEigMode_t    jobz,
                                            int                   m,
                                            int                   n,
                                            double*               A,
                                            int                   lda,
                                            double*               S,
                                            double*               U,
                                            int                   ldu,
                                            double*               V,
                                            int                   ldv,
                                            double*               work,
                                            int                   lwork,
                                            int*                  devInfo,
                                            hipsolverGesvdjInfo_t params,
                                            int                   batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgesvdjBatched(hipsolverDnHandle_t   handle,
                                            hipsolverEigMode_t    jobz,
                                            int                   m,
                                            int                   n,
                                            hipFloatComplex*      A,
                                            int                   lda,
                                            float*                S,
                                            hipFloatComplex*      U,
                                            int                   ldu,
                                            hipFloatComplex*      V,
                                            int                   ldv,
                                            hipFloatComplex*      work,
                                            int                   lwork,
                                            int*                  devInfo,
                                            hipsolverGesvdjInfo_t params,
                                            int                   batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

 hipsolverStatus_t hipsolverDnZgesvdjBatched(hipsolverDnHandle_t   handle,
                                             hipsolverEigMode_t    jobz,
                                             int                   m,
                                             int                   n,
                                             hipDoubleComplex*     A,
                                             int                   lda,
                                             double*               S,
                                             hipDoubleComplex*     U,
                                             int                   ldu,
                                             hipDoubleComplex*     V,
                                             int                   ldv,
                                             hipDoubleComplex*     work,
                                             int                   lwork,
                                             int*                  devInfo,
                                             hipsolverGesvdjInfo_t params,
                                             int                   batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnSgesvdaStridedBatched_bufferSize(hipsolverHandle_t  handle,
                                                hipsolverEigMode_t jobz,
                                                int                rank,
                                                int                m,
                                                int                n,
                                                const float*       A,
                                                int                lda,
                                                long long int      strideA,
                                                const float*       S,
                                                long long int      strideS,
                                                const float*       U,
                                                int                ldu,
                                                long long int      strideU,
                                                const float*       V,
                                                int                ldv,
                                                long long int      strideV,
                                                int*               lwork,
                                                int                batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnDgesvdaStridedBatched_bufferSize(hipsolverHandle_t  handle,
                                                hipsolverEigMode_t jobz,
                                                int                rank,
                                                int                m,
                                                int                n,
                                                const double*      A,
                                                int                lda,
                                                long long int      strideA,
                                                const double*      S,
                                                long long int      strideS,
                                                const double*      U,
                                                int                ldu,
                                                long long int      strideU,
                                                const double*      V,
                                                int                ldv,
                                                long long int      strideV,
                                                int*               lwork,
                                                int                batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnCgesvdaStridedBatched_bufferSize(hipsolverHandle_t      handle,
                                                hipsolverEigMode_t     jobz,
                                                int                    rank,
                                                int                    m,
                                                int                    n,
                                                const hipFloatComplex* A,
                                                int                    lda,
                                                long long int          strideA,
                                                const float*           S,
                                                long long int          strideS,
                                                const hipFloatComplex* U,
                                                int                    ldu,
                                                long long int          strideU,
                                                const hipFloatComplex* V,
                                                int                    ldv,
                                                long long int          strideV,
                                                int*                   lwork,
                                                int                    batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t
    hipsolverDnZgesvdaStridedBatched_bufferSize(hipsolverHandle_t       handle,
                                                hipsolverEigMode_t      jobz,
                                                int                     rank,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* A,
                                                int                     lda,
                                                long long int           strideA,
                                                const double*           S,
                                                long long int           strideS,
                                                const hipDoubleComplex* U,
                                                int                     ldu,
                                                long long int           strideU,
                                                const hipDoubleComplex* V,
                                                int                     ldv,
                                                long long int           strideV,
                                                int*                    lwork,
                                                int                     batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSgesvdaStridedBatched(hipsolverHandle_t  handle,
                                                   hipsolverEigMode_t jobz,
                                                   int                rank,
                                                   int                m,
                                                   int                n,
                                                   const float*       A,
                                                   int                lda,
                                                   long long int      strideA,
                                                   float*             S,
                                                   long long int      strideS,
                                                   float*             U,
                                                   int                ldu,
                                                   long long int      strideU,
                                                   float*             V,
                                                   int                ldv,
                                                   long long int      strideV,
                                                   float*             work,
                                                   int                lwork,
                                                   int*               devInfo,
                                                   double*            hRnrmF,
                                                   int                batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDgesvdaStridedBatched(hipsolverHandle_t  handle,
                                                   hipsolverEigMode_t jobz,
                                                   int                rank,
                                                   int                m,
                                                   int                n,
                                                   const double*      A,
                                                   int                lda,
                                                   long long int      strideA,
                                                   double*            S,
                                                   long long int      strideS,
                                                   double*            U,
                                                   int                ldu,
                                                   long long int      strideU,
                                                   double*            V,
                                                   int                ldv,
                                                   long long int      strideV,
                                                   double*            work,
                                                   int                lwork,
                                                   int*               devInfo,
                                                   double*            hRnrmF,
                                                   int                batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCgesvdaStridedBatched(hipsolverHandle_t      handle,
                                                   hipsolverEigMode_t     jobz,
                                                   int                    rank,
                                                   int                    m,
                                                   int                    n,
                                                   const hipFloatComplex* A,
                                                   int                    lda,
                                                   long long int          strideA,
                                                   float*                 S,
                                                   long long int          strideS,
                                                   hipFloatComplex*       U,
                                                   int                    ldu,
                                                   long long int          strideU,
                                                   hipFloatComplex*       V,
                                                   int                    ldv,
                                                   long long int          strideV,
                                                   hipFloatComplex*       work,
                                                   int                    lwork,
                                                   int*                   devInfo,
                                                   double*                hRnrmF,
                                                   int batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZgesvdaStridedBatched(hipsolverHandle_t       handle,
                                                   hipsolverEigMode_t      jobz,
                                                   int                     rank,
                                                   int                     m,
                                                   int                     n,
                                                   const hipDoubleComplex* A,
                                                   int                     lda,
                                                   long long int           strideA,
                                                   double*                 S,
                                                   long long int           strideS,
                                                   hipDoubleComplex*       U,
                                                   int                     ldu,
                                                   long long int           strideU,
                                                   hipDoubleComplex*       V,
                                                   int                     ldv,
                                                   long long int           strideV,
                                                   hipDoubleComplex*       work,
                                                   int                     lwork,
                                                   int*                    devInfo,
                                                   double*                 hRnrmF,
                                                   int batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSsyevd_bufferSize(hipsolverHandle_t  handle,
                                               hipsolverEigMode_t jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                n,
                                               const float*       A,
                                               int                lda,
                                               const float*       W,
                                               int*               lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDsyevd_bufferSize(hipsolverHandle_t  handle,
                                               hipsolverEigMode_t jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                n,
                                               const double*      A,
                                               int                lda,
                                               const double*      W,
                                               int*               lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCheevd_bufferSize(hipsolverHandle_t      handle,
                                               hipsolverEigMode_t     jobz,
                                               hipsolverFillMode_t      uplo,
                                               int                    n,
                                               const hipFloatComplex* A,
                                               int                    lda,
                                               const float*           W,
                                               int*                   lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZheevd_bufferSize(hipsolverHandle_t       handle,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t       uplo,
                                               int                     n,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               const double*           W,
                                               int*                    lwork) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSsyevd(hipsolverHandle_t  handle,
                                    hipsolverEigMode_t jobz,
                                    hipsolverFillMode_t  uplo,
                                    int                n,
                                    float*             A,
                                    int                lda,
                                    float*             W,
                                    float*             work,
                                    int                lwork,
                                    int*               devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDsyevd(hipsolverHandle_t  handle,
                                    hipsolverEigMode_t jobz,
                                    hipsolverFillMode_t  uplo,
                                    int                n,
                                    double*            A,
                                    int                lda,
                                    double*            W,
                                    double*            work,
                                    int                lwork,
                                    int*               devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCheevd(hipsolverHandle_t  handle,
                                    hipsolverEigMode_t jobz,
                                    hipsolverFillMode_t  uplo,
                                    int                n,
                                    hipFloatComplex*   A,
                                    int                lda,
                                    float*             W,
                                    hipFloatComplex*   work,
                                    int                lwork,
                                    int*               devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZheevd(hipsolverHandle_t  handle,
                                    hipsolverEigMode_t jobz,
                                    hipsolverFillMode_t  uplo,
                                    int                n,
                                    hipDoubleComplex*  A,
                                    int                lda,
                                    double*            W,
                                    hipDoubleComplex*  work,
                                    int                lwork,
                                    int*               devInfo) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCreateSyevjInfo(hipsolverSyevjInfo_t* info) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDestroySyevjInfo(hipsolverSyevjInfo_t info) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXsyevjSetTolerance(hipsolverSyevjInfo_t info,
                                                double               tolerance) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXsyevjSetMaxSweeps(hipsolverSyevjInfo_t info,
                                                int                  max_sweeps) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXsyevjSetSortEig(hipsolverSyevjInfo_t info,
                                              int                  sort_eig) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXsyevjGetResidual(hipsolverDnHandle_t  handle,
                                               hipsolverSyevjInfo_t info,
                                               double*              residual) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXsyevjGetSweeps(hipsolverDnHandle_t  handle,
                                             hipsolverSyevjInfo_t info,
                                             int*                 executed_sweeps) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSsyevj_bufferSize(hipsolverDnHandle_t  handle,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t    uplo,
                                               int                  n,
                                               const float*         A,
                                               int                  lda,
                                               const float*         W,
                                               int*                 lwork,
                                               hipsolverSyevjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDsyevj_bufferSize(hipsolverDnHandle_t  handle,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t    uplo,
                                               int                  n,
                                               const double*        A,
                                               int                  lda,
                                               const double*        W,
                                               int*                 lwork,
                                               hipsolverSyevjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCheevj_bufferSize(hipsolverDnHandle_t    handle,
                                               hipsolverEigMode_t     jobz,
                                               hipsolverFillMode_t      uplo,
                                               int                    n,
                                               const hipFloatComplex* A,
                                               int                    lda,
                                               const float*           W,
                                               int*                   lwork,
                                               hipsolverSyevjInfo_t   params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZheevj_bufferSize(hipsolverDnHandle_t     handle,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t       uplo,
                                               int                     n,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               const double*           W,
                                               int*                    lwork,
                                               hipsolverSyevjInfo_t    params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSsyevj(hipsolverDnHandle_t  handle,
                                    hipsolverEigMode_t   jobz,
                                    hipsolverFillMode_t    uplo,
                                    int                  n,
                                    float*               A,
                                    int                  lda,
                                    float*               W,
                                    float*               work,
                                    int                  lwork,
                                    int*                 devInfo,
                                    hipsolverSyevjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDsyevj(hipsolverDnHandle_t  handle,
                                    hipsolverEigMode_t   jobz,
                                    hipsolverFillMode_t    uplo,
                                    int                  n,
                                    double*              A,
                                    int                  lda,
                                    double*              W,
                                    double*              work,
                                    int                  lwork,
                                    int*                 devInfo,
                                    hipsolverSyevjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCheevj(hipsolverDnHandle_t  handle,
                                    hipsolverEigMode_t   jobz,
                                    hipsolverFillMode_t    uplo,
                                    int                  n,
                                    hipFloatComplex*     A,
                                    int                  lda,
                                    float*               W,
                                    hipFloatComplex*     work,
                                    int                  lwork,
                                    int*                 devInfo,
                                    hipsolverSyevjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZheevj(hipsolverDnHandle_t  handle,
                                    hipsolverEigMode_t   jobz,
                                    hipsolverFillMode_t    uplo,
                                    int                  n,
                                    hipDoubleComplex*    A,
                                    int                  lda,
                                    double*              W,
                                    hipDoubleComplex*    work,
                                    int                  lwork,
                                    int*                 devInfo,
                                    hipsolverSyevjInfo_t params) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSsyevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                      hipsolverEigMode_t   jobz,
                                                      hipsolverFillMode_t    uplo,
                                                      int                  n,
                                                      const float*         A,
                                                      int                  lda,
                                                      const float*         W,
                                                      int*                 lwork,
                                                      hipsolverSyevjInfo_t params,
                                                      int batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDsyevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                      hipsolverEigMode_t   jobz,
                                                      hipsolverFillMode_t    uplo,
                                                      int                  n,
                                                      const double*        A,
                                                      int                  lda,
                                                      const double*        W,
                                                      int*                 lwork,
                                                      hipsolverSyevjInfo_t params,
                                                      int batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCheevjBatched_bufferSize(hipsolverDnHandle_t handle,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverFillMode_t   uplo,
                                                      int                 n,
                                                      const hipFloatComplex* A,
                                                      int                    lda,
                                                      const float*           W,
                                                      int*                   lwork,
                                                      hipsolverSyevjInfo_t params,
                                                      int batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZheevjBatched_bufferSize(hipsolverDnHandle_t handle,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverFillMode_t   uplo,
                                                      int                 n,
                                                      const hipDoubleComplex* A,
                                                      int                     lda,
                                                      const double*           W,
                                                      int*                 lwork,
                                                      hipsolverSyevjInfo_t params,
                                                      int batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnSsyevjBatched(hipsolverDnHandle_t  handle,
                                           hipsolverEigMode_t   jobz,
                                           hipsolverFillMode_t    uplo,
                                           int                  n,
                                           float*               A,
                                           int                  lda,
                                           float*               W,
                                           float*               work,
                                           int                  lwork,
                                           int*                 devInfo,
                                           hipsolverSyevjInfo_t params,
                                           int                  batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnDsyevjBatched(hipsolverDnHandle_t  handle,
                                           hipsolverEigMode_t   jobz,
                                           hipsolverFillMode_t    uplo,
                                           int                  n,
                                           double*              A,
                                           int                  lda,
                                           double*              W,
                                           double*              work,
                                           int                  lwork,
                                           int*                 devInfo,
                                           hipsolverSyevjInfo_t params,
                                           int                  batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnCheevjBatched(hipsolverDnHandle_t  handle,
                                           hipsolverEigMode_t   jobz,
                                           hipsolverFillMode_t    uplo,
                                           int                  n,
                                           hipFloatComplex*     A,
                                           int                  lda,
                                           float*               W,
                                           hipFloatComplex*     work,
                                           int                  lwork,
                                           int*                 devInfo,
                                           hipsolverSyevjInfo_t params,
                                           int                  batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnZheevjBatched(hipsolverDnHandle_t  handle,
                                           hipsolverEigMode_t   jobz,
                                           hipsolverFillMode_t    uplo,
                                           int                  n,
                                           hipDoubleComplex*    A,
                                           int                  lda,
                                           double*              W,
                                           hipDoubleComplex*    work,
                                           int                  lwork,
                                           int*                 devInfo,
                                           hipsolverSyevjInfo_t params,
                                           int                  batch_count) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

#endif

hipsolverStatus_t cusolverDnCreateParams(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDestroyParams(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

typedef void* cusolverSpHandle_t;
typedef void* hipsparseMatDescr_t;

hipsolverStatus_t cusolverSpGetStream(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpSetStream(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZZgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZCgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZYgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZKgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnCCgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnCYgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnCKgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnDDgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnDSgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnDXgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnDHgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnSSgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnSXgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnSHgels_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnZZgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnZCgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnZYgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnZKgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnCCgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnCYgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnCKgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnDDgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnDSgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnDXgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnDHgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnSSgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnSXgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
hipsolverStatus_t cusolverDnSHgels(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZZgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZCgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZYgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZKgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnCCgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnCYgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnCKgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDDgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDSgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDXgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDHgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnSSgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnSXgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnSHgesv_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZZgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZCgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZYgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnZKgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnCCgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnCYgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnCKgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDDgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDSgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDXgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnDHgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnSSgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnSXgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverDnSHgesv(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXsyevd_bufferSize(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t hipsolverDnXsyevd(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpCreate(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpDestroy(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpScsrlsvqr(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpDcsrlsvqr(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpCcsrlsvqr(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpZcsrlsvqr(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpScsrlsvchol(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpDcsrlsvchol(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpCcsrlsvchol(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpZcsrlsvchol(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpScsreigvsi(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpDcsreigvsi(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpCcsreigvsi(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

hipsolverStatus_t cusolverSpZcsreigvsi(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

} // extern "C"

#endif // #ifdef INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H
