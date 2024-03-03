from libc.stdint cimport intptr_t


###############################################################################
# Enum
###############################################################################

cpdef enum:
    CUDNN_NOT_PROPAGATE_NAN = 0
    CUDNN_PROPAGATE_NAN = 1

    CUDNN_TENSOR_NCHW = 0
    CUDNN_TENSOR_NHWC = 1

    CUDNN_OP_TENSOR_ADD = 0
    CUDNN_OP_TENSOR_MUL = 1
    CUDNN_OP_TENSOR_MIN = 2
    CUDNN_OP_TENSOR_MAX = 3

    CUDNN_REDUCE_TENSOR_ADD = 0
    CUDNN_REDUCE_TENSOR_MUL = 1
    CUDNN_REDUCE_TENSOR_MIN = 2
    CUDNN_REDUCE_TENSOR_MAX = 3
    CUDNN_REDUCE_TENSOR_AMAX = 4
    CUDNN_REDUCE_TENSOR_AVG = 5
    CUDNN_REDUCE_TENSOR_NORM1 = 6
    CUDNN_REDUCE_TENSOR_NORM2 = 7

    CUDNN_REDUCE_TENSOR_NO_INDICES = 0
    CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1

    CUDNN_32BIT_INDICES = 0
    CUDNN_64BIT_INDICES = 1
    CUDNN_16BIT_INDICES = 2
    CUDNN_8BIT_INDICES = 3

    # TODO Confirm from miopen team 
    CUDNN_CONVOLUTION = 0
    CUDNN_CROSS_CORRELATION = 1

    CUDNN_SOFTMAX_FAST = 0
    CUDNN_SOFTMAX_ACCURATE = 1
    CUDNN_SOFTMAX_LOG = 2

    CUDNN_SOFTMAX_MODE_INSTANCE = 0
    CUDNN_SOFTMAX_MODE_CHANNEL = 1

    CUDNN_BATCHNORM_PER_ACTIVATION = 0
    CUDNN_BATCHNORM_SPATIAL = 1

    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0

    CUDNN_RNN_RELU = 0
    CUDNN_RNN_TANH = 1
    CUDNN_LSTM = 2
    CUDNN_GRU = 3

    CUDNN_UNIDIRECTIONAL = 0
    CUDNN_BIDIRECTIONAL = 1

    CUDNN_RNN_PADDED_IO_DISABLED = 0
    CUDNN_RNN_PADDED_IO_ENABLED = 1

    CUDNN_LINEAR_INPUT = 0
    CUDNN_SKIP_INPUT = 1

    CUDNN_STATUS_SUCCESS = 0
    
cpdef enum:
    CUDNN_DATA_FLOAT = 1
    CUDNN_DATA_DOUBLE = 6
    CUDNN_DATA_HALF = 0
    
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 5
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 1
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 2
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 3

    CUDNN_POOLING_MAX = 0
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 2
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 1
            
    CUDNN_ACTIVATION_RELU = 3
    CUDNN_ACTIVATION_TANH = 2
    CUDNN_ACTIVATION_CLIPPED_RELU = 7
    CUDNN_ACTIVATION_ELU = 9
    CUDNN_ACTIVATION_IDENTITY = 0
            
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 1
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 2
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 3

###############################################################################
# Class
###############################################################################

cdef class CuDNNAlgoPerf:
    cdef:
        int algo
        int status
        float time
        size_t memory
        int determinism
        int mathType

###############################################################################
# Version
###############################################################################

cpdef size_t getVersion() except? 0
###############################################################################
# Initialization and CUDA cooperation
###############################################################################

cpdef intptr_t create() except? 0
cpdef destroy(intptr_t handle)
cpdef setStream(intptr_t handle, size_t stream)
cpdef size_t getStream(intptr_t handle) except? 0

###############################################################################
# Tensor manipulation
###############################################################################

cpdef size_t createTensorDescriptor() except? 0
cpdef setTensor4dDescriptor(size_t tensorDesc, int format, int dataType,
                            int n, int c, int h, int w)
cpdef setTensor4dDescriptorEx(size_t tensorDesc, int dataType,
                              int n, int c, int h, int w, int nStride,
                              int cStride, int hStride, int wStride)
cpdef tuple getTensor4dDescriptor(size_t tensorDesc)
cpdef destroyTensorDescriptor(size_t tensorDesc)

###############################################################################
# Activation
###############################################################################

cpdef size_t createActivationDescriptor() except? 0
cpdef setActivationDescriptor(
    size_t activationDesc, int mode, int reluNanOpt, double reluCeiling)
cpdef destroyActivationDescriptor(size_t activationDesc)
cpdef softmaxForward(
    intptr_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
    size_t srcData, size_t beta, size_t dstDesc, size_t dstData)
cpdef softmaxBackward(
    intptr_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
    size_t srcData, size_t srcDiffDesc, size_t srcDiffData, size_t beta,
    size_t destDiffDesc, size_t destDiffData)
cpdef activationForward_v4(
    intptr_t handle, size_t activationDesc, size_t alpha, size_t srcDesc,
    size_t srcData, size_t beta, size_t dstDesc, size_t dstData)
cpdef activationBackward_v4(
    intptr_t handle, size_t activationDesc, size_t alpha, size_t srcDesc,
    size_t srcData, size_t srcDiffDesc, size_t srcDiffData,
    size_t destDesc, size_t destData, size_t beta, size_t destDiffDesc,
    size_t destDiffData)

