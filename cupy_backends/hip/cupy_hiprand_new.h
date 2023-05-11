#ifndef INCLUDE_GUARD_HIP_CUPY_HIPRAND_NEW_H
#define INCLUDE_GUARD_HIP_CUPY_HIPRAND_NEW_H

#include <hiprand/hiprand.h>

extern "C" {

typedef enum {} hiprandOrdering_t;

//curandRngType_t convert_hiprandRngType(curandRngType_t t) {
//    switch(static_cast<int>(t)) {
//    case 100: return HIPRAND_RNG_PSEUDO_DEFAULT;
//    case 101: return HIPRAND_RNG_PSEUDO_XORWOW;
//    case 121: return HIPRAND_RNG_PSEUDO_MRG32K3A;
//    case 141: return HIPRAND_RNG_PSEUDO_MTGP32;
//    case 142: return HIPRAND_RNG_PSEUDO_MT19937;
//    case 161: return HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
//    case 200: return HIPRAND_RNG_QUASI_DEFAULT;
//    case 201: return HIPRAND_RNG_QUASI_SOBOL32;
//    case 202: return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
//    case 203: return HIPRAND_RNG_QUASI_SOBOL64;
//    case 204: return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
//    }
//    return HIPRAND_RNG_TEST;
//}

hiprandStatus_t hiprandSetGeneratorOrdering(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

hiprandStatus_t hiprandGenerateLongLong(...) {
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_HIP_CUPY_HIPRAND_NEW_H
