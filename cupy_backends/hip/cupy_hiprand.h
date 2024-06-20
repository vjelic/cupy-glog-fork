#ifndef INCLUDE_GUARD_HIP_CUPY_HIPRAND_H
#define INCLUDE_GUARD_HIP_CUPY_HIPRAND_H

#include <hiprand/hiprand.h>

extern "C" {

#if HIP_VERSION < 60241132
typedef enum {} hiprandOrdering_t;

hiprandStatus_t hiprandSetGeneratorOrdering(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}
#endif

#if HIP_VERSION < 50530201
hiprandStatus_t hiprandGenerateLongLong(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}
#endif

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPRAND_H
