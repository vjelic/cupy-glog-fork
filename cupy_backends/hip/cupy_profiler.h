#ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
#define INCLUDE_GUARD_HIP_CUPY_PROFILER_H

#include "cupy_hip_common.h"
#include "roctracer/roctracer_ext.h"

extern "C" {

typedef enum {} cudaOutputMode_t;

hipError_t hipProfilerInitialize(...) {
    return hipSuccess;
}

void hipProfilerStart() {
    return roctracer_start();
}

void hipProfilerStop() {
    return roctracer_stop();
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_PROFILER_H
