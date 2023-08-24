#ifndef INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H
#define INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H

#include "cupy_hip.h"
#include "cupy_hipblas.h"
#include <stdexcept>  // for gcc 10.0

extern "C" {

static hipsolverFillMode_t convert_hipsolver_fill(hipsolverFillMode_t mode) {
    return static_cast<hipsolverFillMode_t>(static_cast<int>(mode) + 121);
}

static hipsolverOperation_t convert_hipsolver_operation(hipsolverOperation_t op) {
    return static_cast<hipsolverOperation_t>(static_cast<int>(op) + 111);
}

static hipsolverSideMode_t convert_hipsolver_side(hipsolverSideMode_t mode) {
    return static_cast<hipsolverSideMode_t>(static_cast<int>(mode) + 141);
}

hipsolverStatus_t cusolverGetProperty(hipLibraryPropertyType_t type, int* val) {
    switch(type) {
        case MAJOR_VERSION: { *val = hipsolverVersionMajor; break; }
        case MINOR_VERSION: { *val = hipsolverVersionMinor; break; }
        case PATCH_LEVEL:   { *val = hipsolverVersionPatch; break; }
        default: throw std::runtime_error("invalid type");
    }
    return HIPSOLVER_STATUS_SUCCESS;
}

typedef enum hipsolverDnParams_t {};

hipsolverStatus_t cusolverDnCreateParams(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED; 
}

hipsolverStatus_t cusolverDnDestroyParams(...) {
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}

//typedef enum{} cusolverEigType_t;
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

/* The following APIs need updating to enum values.
   The below APIs can be removed after cublas is implemented in hipify flow.
*/

hipsolverStatus_t hipsolverDnSpotrf_bufferSize(hipsolverDnHandle_t handle,
                                             hipsolverFillMode_t uplo,
                                             int n,
                                             float *A,
                                             int lda,
                                             int *Lwork) {
    return hipsolverDnSpotrf_bufferSize(handle, convert_hipsolver_fill(uplo),
                                        n, A, lda, Lwork);
}

hipsolverStatus_t hipsolverDnDpotrf_bufferSize(hipsolverDnHandle_t handle,
                                             hipsolverFillMode_t uplo,
                                             int n,
                                             double *A,
                                             int lda,
                                             int *Lwork) {
    return hipsolverDnDpotrf_bufferSize(handle, convert_hipsolver_fill(uplo),
                                        n, A, lda, Lwork); 
}

hipsolverStatus_t hipsolverDnCpotrf_bufferSize(hipsolverDnHandle_t handle,
                                             hipsolverFillMode_t uplo,
                                             int n,
                                             hipFloatComplex *A,
                                             int lda,
                                             int *Lwork) {
    return hipsolverDnCpotrf_bufferSize(handle, convert_hipsolver_fill(uplo),
                                        n, A, lda, Lwork);
}

hipsolverStatus_t hipsolverDnZpotrf_bufferSize(hipsolverDnHandle_t handle,
                                             hipsolverFillMode_t uplo,
                                             int n,
                                             hipDoubleComplex *A,
                                             int lda,
                                             int *Lwork) {
    return hipsolverDnZpotrf_bufferSize(handle, convert_hipsolver_fill(uplo),
                                        n, A, lda, Lwork);
}

hipsolverStatus_t hipsolverDnSpotrf(hipsolverDnHandle_t handle,
                                  hipsolverFillMode_t uplo,
                                  int n,
                                  float *A,
                                  int lda,
                                  float *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    return hipsolverDnSpotrf(handle, convert_hipsolver_fill(uplo),
                             n, A, lda, Workspace, Lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDpotrf(hipsolverDnHandle_t handle,
                                  hipsolverFillMode_t uplo,
                                  int n,
                                  double *A,
                                  int lda,
                                  double *Workspace,
                                  int Lwork,
                                  int *devInfo ) {
    return hipsolverDnDpotrf(handle, convert_hipsolver_fill(uplo),
                             n, A, lda, Workspace, Lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCpotrf(hipsolverDnHandle_t handle,
                                  hipsolverFillMode_t uplo,
                                  int n,
                                  hipFloatComplex *A,
                                  int lda,
                                  hipFloatComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    return hipsolverDnCpotrf(handle, convert_hipsolver_fill(uplo),
                             n, A, lda, Workspace, Lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZpotrf(hipsolverDnHandle_t handle,
                                  hipsolverFillMode_t uplo,
                                  int n,
                                  hipDoubleComplex *A,
                                  int lda,
                                  hipDoubleComplex *Workspace,
                                  int Lwork,
                                  int *devInfo) {
    return hipsolverDnZpotrf(handle, convert_hipsolver_fill(uplo),
                             n, A, lda, Workspace, Lwork, devInfo);
}


} // extern "C" 

#endif // #ifdef INCLUDE_GUARD_HIP_CUPY_ROCSOLVER_H
