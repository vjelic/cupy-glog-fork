#ifndef INCLUDE_GUARD_HIP_CUPY_HIPRTC_H
#define INCLUDE_GUARD_HIP_CUPY_HIPRTC_H

#include <hip/hiprtc.h>

extern "C" {

#if HIP_VERSION < 50631061 

hiprtcResult hiprtcGetBitcode(hiprtcProgram prog, char *bitcode) {
    return HIPRTC_ERROR_COMPILATION;
}

hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog, size_t *bitcode_size) {
    return HIPRTC_ERROR_COMPILATION;
}
#endif

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPRTC_H
