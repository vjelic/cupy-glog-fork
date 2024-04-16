#ifndef INCLUDE_GUARD_HIP_CUPY_RUNTIME_H
#define INCLUDE_GUARD_HIP_CUPY_RUNTIME_H

#include <hip/hip_runtime_api.h>
#include "cupy_hip_common.h"

extern "C" {

bool hip_environment = true;

// Stream and Event
#if HIP_VERSION >= 40300000
typedef hipStreamCaptureMode cudaStreamCaptureMode;
typedef hipStreamCaptureStatus cudaStreamCaptureStatus;
#else
enum cudaStreamCaptureMode {};
enum cudaStreamCaptureStatus {};
#endif

// CUDA Graph
#if 0 < HIP_VERSION < 40300000
cudaError_t cudaGraphInstantiate(
	cudaGraphExec_t* pGraphExec,
	cudaGraph_t graph,
	cudaGraphNode_t* pErrorNode,
	char* pLogBuffer,
	size_t bufferSize) {
    return hipErrorUnknown;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    return hipErrorUnknown;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    return hipErrorUnknown;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    return hipErrorUnknown;
}

#elif 0 < HIP_VERSION < 50300000
cudaError_t cudaGraphUpload(...) {
    return hipErrorUnknown;
}
#endif

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_RUNTIME_H
