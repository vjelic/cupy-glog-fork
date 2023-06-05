#ifndef INCLUDE_GUARD_HIP_CUPY_HIPRAND_NEW_H
#define INCLUDE_GUARD_HIP_CUPY_HIPRAND_NEW_H

#include <hiprand/hiprand.h>

extern "C" {

typedef enum {} hiprandOrdering_t;

hiprandStatus_t hiprandSetGeneratorOrdering(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPRAND_NEW_H
