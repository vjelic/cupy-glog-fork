# distutils: language = c++

"""Thin wrapper of MIOpen."""
cimport cython  # NOQA
from libcpp cimport vector
from libcpp cimport bool
from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module

from libc.stdint cimport intptr_t
###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_miopen.h' nogil:
    # Types
    ctypedef int ActivationMode 'miopenActivationMode_t'
    ctypedef int BatchNormMode 'miopenBatchNormMode_t'
    ctypedef int ConvolutionBwdDataAlgo 'miopenConvBwdDataAlgorithm_t'
    ctypedef int ConvolutionBwdFilterAlgo 'miopenConvBwdWeightsAlgorithm_t'
    ctypedef int ConvolutionFwdAlgo 'miopenConvFwdAlgorithm_t'
    ctypedef int ConvolutionMode 'miopenConvolutionMode_t'
    ctypedef struct ConvolutionFwdAlgoPerf 'miopenConvAlgoPerf_t':
        int fwd_algo
        int bwd_weights_algo
        int bwd_data_algo
        float time
        size_t memory
    ctypedef int DataType 'miopenDataType_t'
    ctypedef int DirectionMode 'miopenRNNDirectionMode_t'
    ctypedef int NanPropagation 'miopenNanPropagation_t'
    ctypedef int PoolingMode 'miopenPoolingMode_t'
    ctypedef int RNNInputMode 'miopenRNNInputMode_t'
    ctypedef int CTCLossAlgo 'miopenCTCLossAlgo_t'
    ctypedef int RNNMode 'miopenRNNMode_t'
    ctypedef int RNNAlgo 'miopenRNNAlgo_t'
    ctypedef int RNNDataLayout 'miopenRNNBaseLayout_t'
    ctypedef int RNNPaddingMode 'miopenRNNPaddingMode_t'
    ctypedef int SoftmaxAlgorithm 'miopenSoftmaxAlgorithm_t'
    ctypedef int SoftmaxMode 'miopenSoftmaxMode_t'
    ctypedef int Status 'miopenStatus_t'
    ctypedef int TensorFormat 'miopenTensorLayout_t'
    ctypedef int OpTensorOp 'miopenTensorOp_t'
    ctypedef int RNGType_t 'miopenRNGType_t' 
    ctypedef int ReduceTensorOp 'miopenReduceTensorOp_t'
    ctypedef int ReduceTensorIndices 'miopenReduceTensorIndices_t'
    ctypedef int IndicesType 'miopenIndicesType_t'
    ctypedef void* ActivationDescriptor 'miopenActivationDescriptor_t'
    ctypedef void* ConvolutionDescriptor 'miopenConvolutionDescriptor_t'
    ctypedef void* DropoutDescriptor 'miopenDropoutDescriptor_t'
    ctypedef void* Handle 'miopenHandle_t'
    ctypedef void* PoolingDescriptor 'miopenPoolingDescriptor_t'
    ctypedef void* CTCLossDescriptor 'miopenCTCLossDescriptor_t'
    ctypedef void* RNNDescriptor 'miopenRNNDescriptor_t'
    ctypedef void* RNNDataDescriptor 'miopenRNNDataDescriptor_t'
    ctypedef void* TensorDescriptor 'miopenTensorDescriptor_t'
    ctypedef void* FilterDescriptor 'miopenTensorDescriptor_t'
    ctypedef void* OpTensorDescriptor 'miopenTensorDescriptor_t'
    ctypedef void* ReduceTensorDescriptor 'miopenReduceTensorDescriptor_t'
    ctypedef void* Stream 'miopenAcceleratorQueue_t'
    # Error handling
    const char* miopenGetErrorString(Status status)
        
    # Version
    #size_t miopenGetVersion()
        
    # Runtime error checking
    #int cudnnQueryRuntimeError(Handle handle, Status *rstatus,
    #                           ErrQueryMode mode, RuntimeTag *tag)
        
    # Initialization and CUDA cooperation
    int miopenCreate(Handle* handle)
    int miopenDestroy(Handle handle)
    int miopenSetStream(Handle handle, driver.Stream stream)
    int miopenGetStream(Handle handle, driver.Stream* stream)
        
    # Tensor manipulation
    int miopenCreateTensorDescriptor(TensorDescriptor* descriptor)
    int miopenSet4dTensorDescriptor(
        TensorDescriptor tensorDesc, 
        DataType dataType, int n, int c, int h, int w)
    int miopenSet4dTensorDescriptorEx(
        TensorDescriptor tensorDesc, DataType dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride)
    int miopenSetTensorDescriptor(
        TensorDescriptor tensorDesc, DataType dataType,
        int nbDims, const int filterDimA[], const int* stride)
    int miopenGetTensorDescriptor(
        FilterDescriptor wDesc, DataType* dataType,
        int* nbDims, int filterDimA[], int* stride)
    int miopenGet4dTensorDescriptor(
        TensorDescriptor tensorDesc, DataType* dataType,
        int* n, int* c, int* h, int* w,
        int* nStride, int* cStride, int* hStride, int* wStride)
    int miopenDestroyTensorDescriptor(TensorDescriptor tensorDesc)
        
    # Tensor operations
    int miopenOpTensor(
        Handle handle, OpTensorDescriptor opTensorDesc, void* alpha1,
        TensorDescriptor aDesc, void* A, void* alpha2,
        TensorDescriptor bDesc, void* B, void* beta,
        TensorDescriptor cDesc, void* C)
        
    # Tensor reductions
    int miopenCreateReduceTensorDescriptor(
        ReduceTensorDescriptor* reduceTensorDesc)
    int miopenSetReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc, ReduceTensorOp reduceTensorOp,
        DataType reduceTensorCompType, NanPropagation reduceTensorNanOpt,
        ReduceTensorIndices reduceTensorIndices,
        IndicesType reduceTensorIndicesType)
    int miopenGetReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc,
        ReduceTensorOp* reduceTensorOp, DataType* reduceTensorCompType,
        NanPropagation* reduceTensorNanOpt,
        ReduceTensorIndices* reduceTensorIndices,
        IndicesType* reduceTensorIndicesType)
    int miopenDestroyReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc)
    int miopenGetReductionIndicesSize(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc,
        TensorDescriptor aDesc, TensorDescriptor cDesc, size_t* sizeInBytes)
    int miopenGetReductionWorkspaceSize(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc,
        TensorDescriptor aDesc, TensorDescriptor cDesc, size_t* sizeInBytes)
    int miopenReduceTensor(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc, void* indices,
        size_t indicesSizeInBytes, void* workspace,
        size_t workspaceSizeInBytes, void* alpha, TensorDescriptor aDesc,
        void* A, void* beta, TensorDescriptor cDesc, void* c)
    int miopenSetTensor(
        Handle handle, TensorDescriptor yDesc, void* y, void* valuePtr)
    int miopenScaleTensor(
        Handle handle, TensorDescriptor yDesc, void* y, void* alpha)
        
    # Filter manipulation
        
    # Convolution
    int miopenCreateConvolutionDescriptor(ConvolutionDescriptor* convDesc)
    int miopenSetConvolutionGroupCount(
        ConvolutionDescriptor convDesc, int groupCount)
    int miopenGetConvolutionGroupCount(
        ConvolutionDescriptor convDesc, int *groupCount)
    int miopenInitConvolutionDescriptor(ConvolutionDescriptor convDesc,
        ConvolutionMode mode, int pad_h, int pad_w, int stride_h, int stride_w,
        int dilation_h, int dilation_w)
    int miopenInitConvolutionNdDescriptor(ConvolutionDescriptor conDesc, int spatialDim,
        const int* padA, const int* strideA,const int* dilationA, ConvolutionMode mode)
    int miopenDestroyConvolutionDescriptor(ConvolutionDescriptor conDesc)
    int miopenConvolutionForwardGetWorkSpaceSize(
        Handle handle, TensorDescriptor srcDesc,
        TensorDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc,
        size_t* sizeInBytes)
    int miopenConvolutionBackwardDataGetWorkSpaceSize(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        size_t* sizeInBytes)
    int miopenFindConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor xDesc, void* x,
        FilterDescriptor wDesc, void* w, ConvolutionDescriptor convDesc,
        TensorDescriptor yDesc, void* y, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionFwdAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes, bool exhaustiveSearch)
    int miopenConvolutionForward(
        Handle handle, void* alpha, TensorDescriptor srcDesc,
        void* srcData, FilterDescriptor filterDesc, void* filterData,
        ConvolutionDescriptor convDesc, ConvolutionFwdAlgo algo,
        void* beta, TensorDescriptor destDesc, void* destData, 
        void* workSpace, size_t workSpaceSizeInBytes)

    # Pooling
    int miopenCreatePoolingDescriptor(PoolingDescriptor* desc)
    int miopenDestroyPoolingDescriptor(PoolingDescriptor poolingDesc)
    # Batch Normalization
    int miopenDeriveBNTensorDescriptor(
        TensorDescriptor derivedBnDesc, TensorDescriptor xDesc,
        BatchNormMode mode)
    int miopenBatchNormalizationForwardTraining(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, double exponentialAverageFactor,
        void* resultRunningMean, void* resultRunningVariance,
        double epsilon, void* resultSaveMean,
        void* resultSaveInvVariance)
    int miopenBatchNormalizationForwardInference(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, void* estimatedMean, void* estimatedVariance,
        double epsilon)
    int miopenBatchNormalizationBackward(
        Handle handle, BatchNormMode mode,
        void* alphaDataDiff, void* betaDataDiff,
        void* alphaParamDiff, void* betaParamDiff,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy,
        TensorDescriptor dxDesc, void* dx,
        TensorDescriptor dBnScaleBiasDesc, void* bnScale,
        void* dBnScaleResult, void* dBnBiasResult,
        double epsilon, void* savedMean, void* savedInvVariance)
        
        
    # Activation
    int miopenCreateActivationDescriptor(
        ActivationDescriptor* activationDesc)
    int miopenSetActivationDescriptor(
        ActivationDescriptor activationDesc, ActivationMode mode, double activAlpha,
        double activBeta,
        double activGamma)
    int miopenDestroyActivationDescriptor(
        ActivationDescriptor activationDesc)
    int miopenSoftmaxForward(
        Handle handle, 
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        void* beta, TensorDescriptor dstDesc, void* dstData)
    int miopenSoftmaxBackward(
        Handle handle, 
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)
    int miopenActivationForward(
        Handle handle, ActivationDescriptor activationDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int miopenActivationBackward(
        Handle handle, ActivationDescriptor activationDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)
        
        
    # Dropout
    int miopenCreateDropoutDescriptor(DropoutDescriptor* desc)
    int miopenDestroyDropoutDescriptor(DropoutDescriptor dropoutDesc)
    int miopenDropoutGetStatesSize(Handle handle, size_t* sizeInBytes)
    int miopenDropoutGetReserveSpaceSize(
        TensorDescriptor xDesc, size_t* sizeInBytes)
    int miopenSetDropoutDescriptor(
        DropoutDescriptor dropoutDesc, Handle handle, float dropout,
        void* states, size_t stateSizeInBytes, unsigned long long seed, 
        bool use_mask, bool state_evo, RNGType_t rng_mode)
    int miopenDropoutForward(
        Handle handle, DropoutDescriptor dropoutDesc, TensorDescriptor noise_shape,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor dstDesc, void* dstData,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)    
    # CTC
    int miopenCreateCTCLossDescriptor(CTCLossDescriptor* ctcLossDesc)
    int miopenDestroyCTCLossDescriptor(CTCLossDescriptor ctcLossDesc)
    int miopenGetCTCLossWorkspaceSize(
        Handle handle, TensorDescriptor probsDesc,
        TensorDescriptor gradientsDesc, int* labels,
        int* labelLengths, int* inputLengths, CTCLossAlgo algo,
        CTCLossDescriptor ctcLossDesc, size_t* sizeInBytes)
    int miopenCTCLoss(
        Handle handle, TensorDescriptor probsDesc,
        void* probs, int* labels, int* labelLengths, int* inputLengths,
        void* costs, TensorDescriptor gradientsDesc, void* gradients,
        CTCLossAlgo algo, CTCLossDescriptor ctcLossDesc,
        void* workspace, size_t workSpaceSizeInBytes)
    # RNN
    int miopenCreateRNNDescriptor(RNNDescriptor* rnnDesc)
    int miopenDestroyRNNDescriptor(RNNDescriptor rnnDesc)
    int miopenGetRNNWorkspaceSize(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, size_t* sizeInBytes)
    int miopenGetRNNTrainingReserveSize(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, size_t* sizeInBytes)
    int miopenGetRNNParamsSize(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor xDesc,
        size_t* sizeInBytes, DataType dataType)
    int miopenRNNForwardInference(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc,
        void* x, TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc,
        void* cx, FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc,
        void* y, TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc,
        void* cy, void* workspace, size_t workSpaceSizeInBytes)
    int miopenRNNForwardTraining(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, void* x,
        TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc, void* cx,
        FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc, void* y,
        TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes)

cdef class CuDNNAlgoPerf:

    def __init__(self, algo, status, time, memory, determinism, mathType):
        self.algo = algo
        self.status = status
        self.time = time
        self.memory = memory
        self.determinism = determinism
        self.mathType = mathType
############################################################################
# Error handling
###############################################################################

class CuDNNError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        IF CUPY_HIP_VERSION != 0:
            msg = miopenGetErrorString(<Status>status)
        ELSE:
            msg = cudnnGetErrorString(<Status>status)
        super(CuDNNError, self).__init__(
            'cuDNN Error: {}'.format(msg.decode()))
        self._infos = []

    def add_info(self, info):
        assert isinstance(info, str)
        self._infos.append(info)

    def add_infos(self, infos):
        assert isinstance(infos, list)
        self._infos.extend(infos)

    def __str__(self):
        base = super(CuDNNError, self).__str__()
        return base + ''.join(
            '\n  ' + info for info in self._infos)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CuDNNError(status)


###############################################################################
# Build-time version
###############################################################################

def get_build_version():
    IF CUPY_HIP_VERSION != 0:
        return CUPY_HIP_VERSION
    ELSE:
        return CUDNN_VERSION


###############################################################################
# Version
###############################################################################

cpdef size_t getVersion() except? 0:
    IF CUPY_HIP_VERSION != 0:
        return CUPY_HIP_VERSION
    ELSE:
        return cudnnGetVersion()
    

IF CUPY_HIP_VERSION == 0:
    ###############################################################################
    # Runtime error checking
    ###############################################################################
    
    cpdef queryRuntimeError(intptr_t handle, int mode):
        cdef Status rstatus
        with nogil:
            status = cudnnQueryRuntimeError(<Handle>handle, &rstatus,
                                            <ErrQueryMode>mode, <RuntimeTag*>0)
        check_status(status)
        return rstatus


###############################################################################
# Initialization and CUDA cooperation
###############################################################################

cpdef intptr_t create() except? 0:
    cdef Handle handle
    with nogil:
        IF CUPY_HIP_VERSION != 0:
            status = miopenCreate(&handle)
        ELSE:
            status = cudnnCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    with nogil:
        IF CUPY_HIP_VERSION != 0:
            status = miopenDestroy(<Handle>handle)
        ELSE:
            status = cudnnDestroy(<Handle>handle)
    check_status(status)
    

cpdef setStream(intptr_t handle, size_t stream):
    # TODO(leofang): The support of stream capture is not mentioned at all in
    # the cuDNN docs (as of CUDA 11.5), so we disable this functionality.
    if not runtime._is_hip_environment and runtime.streamIsCapturing(stream):
        raise NotImplementedError(
            'calling cuDNN API during stream capture is currently '
            'unsupported')
    IF CUPY_HIP_VERSION != 0:
        status = miopenSetStream(<Handle>handle, <Stream>stream)
    ELSE:
        status = cudnnSetStream(<Handle>handle, <driver.Stream>stream)
    check_status(status)


cpdef size_t getStream(intptr_t handle) except? 0:
    IF CUPY_HIP_VERSION != 0:
        cdef Stream stream
        status = miopenGetStream(<Handle>handle, &stream)  
    ELSE:        
        cdef driver.Stream stream
        status = cudnnGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


cdef _setStream(intptr_t handle):
    """Set current stream"""
    setStream(handle, stream_module.get_current_stream_ptr())


###############################################################################
# Tensor manipulation
###############################################################################

cpdef size_t createTensorDescriptor() except? 0:
    cdef TensorDescriptor descriptor
    IF CUPY_HIP_VERSION != 0:
        status = miopenCreateTensorDescriptor(&descriptor)
    ELSE:
        status = cudnnCreateTensorDescriptor(&descriptor)
    check_status(status)
    return <size_t>descriptor


cpdef setTensor4dDescriptor(size_t tensorDesc, int format, int dataType,
                            int n, int c, int h, int w):
    IF CUPY_HIP_VERSION != 0:
        status = miopenSet4dTensorDescriptor(
            <TensorDescriptor>tensorDesc,
            <DataType>dataType, n, c, h, w)
    ELSE:
        status = cudnnSetTensor4dDescriptor(
            <TensorDescriptor>tensorDesc, <TensorFormat>format,
            <DataType>dataType, n, c, h, w)
    check_status(status)


cpdef setTensor4dDescriptorEx(size_t tensorDesc, int dataType,
                              int n, int c, int h, int w, int nStride,
                              int cStride, int hStride, int wStride):
    IF CUPY_HIP_VERSION != 0:
        status = miopenSet4dTensorDescriptorEx(
            <TensorDescriptor>tensorDesc, <DataType>dataType, n, c, h, w,
            nStride, cStride, hStride, wStride)
    ELSE:
        status = cudnnSetTensor4dDescriptorEx(
            <TensorDescriptor>tensorDesc, <DataType>dataType, n, c, h, w,
            nStride, cStride, hStride, wStride)
    check_status(status)


cpdef tuple getTensor4dDescriptor(size_t tensorDesc):
    cdef DataType dataType
    cdef int n, c, h, w, nStride, cStride, hStride, wStride
    IF CUPY_HIP_VERSION != 0:
        status = miopenGet4dTensorDescriptor(
            <TensorDescriptor>tensorDesc, &dataType,
            &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride)
    ELSE:
        status = cudnnGetTensor4dDescriptor(
            <TensorDescriptor>tensorDesc, &dataType,
            &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride)
    check_status(status)
    return dataType, n, c, h, w, nStride, cStride, hStride, wStride


cpdef destroyTensorDescriptor(size_t tensorDesc):
    IF CUPY_HIP_VERSION != 0:
        status = miopenDestroyTensorDescriptor(<TensorDescriptor>tensorDesc)
    ELSE:
        status = cudnnDestroyTensorDescriptor(<TensorDescriptor>tensorDesc)
    check_status(status)

###############################################################################
# Activation
###############################################################################

cpdef size_t createActivationDescriptor() except? 0:
    cdef ActivationDescriptor activationDesc
    IF CUPY_HIP_VERSION != 0:
        status = miopenCreateActivationDescriptor(&activationDesc)
    ELSE:
        status = cudnnCreateActivationDescriptor(&activationDesc)
    check_status(status)
    return <size_t>activationDesc


cpdef setActivationDescriptor(
        size_t activationDesc, int mode, int reluNanOpt, double reluCeiling):
    IF CUPY_HIP_VERSION != 0:
        status = miopenSetActivationDescriptor(
            <ActivationDescriptor>activationDesc, <ActivationMode>mode, 1.0, 0.0, 0.0)
    ELSE:
        status = cudnnSetActivationDescriptor(
            <ActivationDescriptor>activationDesc, <ActivationMode>mode,
            <NanPropagation>reluNanOpt, reluCeiling)
    check_status(status)


cpdef destroyActivationDescriptor(size_t activationDesc):
    IF CUPY_HIP_VERSION != 0:
        status = miopenDestroyActivationDescriptor(
            <ActivationDescriptor>activationDesc)
    ELSE:    
        status = cudnnDestroyActivationDescriptor(
            <ActivationDescriptor>activationDesc)
    check_status(status)


cpdef softmaxForward(
        intptr_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData):
    _setStream(handle)
    with nogil:
        IF CUPY_HIP_VERSION != 0:
            status = miopenSoftmaxForward(
                <Handle>handle, <void*>alpha, <TensorDescriptor>srcDesc, <void*>srcData,
                <void*>beta, <TensorDescriptor>dstDesc, <void*>dstData)
        ELSE:
            status = cudnnSoftmaxForward(
                <Handle>handle, <SoftmaxAlgorithm>algorithm, <SoftmaxMode>mode,
                <void*>alpha, <TensorDescriptor>srcDesc, <void*>srcData,
                <void*>beta, <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef softmaxBackward(
        intptr_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData, size_t beta,
        size_t destDiffDesc, size_t destDiffData):
    _setStream(handle)
    with nogil:
        IF CUPY_HIP_VERSION != 0:
            status = miopenSoftmaxBackward(
                <Handle>handle, <void*>alpha, <TensorDescriptor>srcDesc, <void*>srcData,
                <TensorDescriptor>srcDiffDesc, <void*>srcDiffData, <void*>beta,
                <TensorDescriptor>destDiffDesc, <void*>destDiffData)
        ELSE:
            status = cudnnSoftmaxBackward(
                <Handle>handle, <SoftmaxAlgorithm>algorithm, <SoftmaxMode>mode,
                <void*>alpha, <TensorDescriptor>srcDesc, <void*>srcData,
                <TensorDescriptor>srcDiffDesc, <void*>srcDiffData, <void*>beta,
                <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)


cpdef activationForward_v4(
        intptr_t handle, size_t activationDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData):
    _setStream(handle)
    with nogil:
        IF CUPY_HIP_VERSION != 0:
            status = miopenActivationForward(
                <Handle>handle, <ActivationDescriptor>activationDesc, <void*>alpha,
                <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
                <TensorDescriptor>dstDesc, <void*>dstData)
        ELSE:
            status = cudnnActivationForward_v4(
                <Handle>handle, <ActivationDescriptor>activationDesc, <void*>alpha,
                <TensorDescriptor>srcDesc, <void*>srcData, <void*>beta,
                <TensorDescriptor>dstDesc, <void*>dstData)
    check_status(status)


cpdef activationBackward_v4(
        intptr_t handle, size_t activationDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData,
        size_t destDesc, size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData):
    _setStream(handle)
    with nogil:
        IF CUPY_HIP_VERSION != 0:
            status = miopenActivationBackward(
                <Handle>handle, <ActivationDescriptor>activationDesc, <void*>alpha,
                <TensorDescriptor>srcDesc, <void*>srcData,
                <TensorDescriptor>srcDiffDesc, <void*>srcDiffData,
                <TensorDescriptor>destDesc, <void*>destData, <void*>beta,
                <TensorDescriptor>destDiffDesc, <void*>destDiffData)
        ELSE:
            status = cudnnActivationBackward_v4(
                <Handle>handle, <ActivationDescriptor>activationDesc, <void*>alpha,
                <TensorDescriptor>srcDesc, <void*>srcData,
                <TensorDescriptor>srcDiffDesc, <void*>srcDiffData,
                <TensorDescriptor>destDesc, <void*>destData, <void*>beta,
                <TensorDescriptor>destDiffDesc, <void*>destDiffData)
    check_status(status)

###############################################################################
# Filter manipulation
###############################################################################

cpdef size_t createFilterDescriptor() except? 0:
    IF CUPY_HIP_VERSION != 0:
        cdef TensorDescriptor desc
        status = miopenCreateTensorDescriptor(&desc)
    ELSE:
        cdef FilterDescriptor desc
        status = cudnnCreateFilterDescriptor(&desc)
    check_status(status)
    return <size_t>desc


cpdef setFilter4dDescriptor_v4(
        size_t filterDesc, int dataType,
        int format, int k, int c, int h, int w):
    IF CUPY_HIP_VERSION != 0:
        status = miopenSet4dTensorDescriptor(
                <TensorDescriptor>filterDesc, <DataType> dataType,
                 k, c, h, w)
    ELSE:
        status = cudnnSetFilter4dDescriptor_v4(
                <FilterDescriptor>filterDesc, <DataType> dataType,
                <TensorFormat> format, k, c, h, w)
    check_status(status)


cpdef setFilterNdDescriptor_v4(
        size_t filterDesc, int dataType,
        int format, int nbDims, size_t filterDimA):
    IF CUPY_HIP_VERSION != 0:
        status = miopenSetTensorDescriptor(
                <TensorDescriptor>filterDesc, <DataType>dataType,
                nbDims, <int*>filterDimA, NULL) #TODO miopenSetTensorDescriptor takes stride as input now set to NULL, confirm the value of stride 
    ELSE:
        status = cudnnSetFilterNdDescriptor_v4(
                <FilterDescriptor>filterDesc, <DataType>dataType,
                <TensorFormat>format, nbDims, <int*>filterDimA)
    check_status(status)

"""
cpdef getFilterNdDescriptor(size_t wDesc, int nbDimsRequested):
    cdef DataType dataType
    cdef TensorFormat format
    cdef int nbDims
    cdef vector.vector[int] filterDimA
    filterDimA.resize(nbDimsRequested)
    IF CUPY_HIP_VERSION != 0:
        status = miopenGetTensorDescriptor(
                <TensorDescriptor>wDesc, &dataType,
                &nbDims, filterDimA.data(), NULL)
    ELSE:
        status = cudnnGetFilterNdDescriptor_v4(
                <FilterDescriptor>wDesc, nbDimsRequested, &dataType,
                &format, &nbDims, filterDimA.data())
    check_status(status)
    return dataType, format, nbDims, tuple(filterDimA)
"""

cpdef destroyFilterDescriptor(size_t filterDesc):
    IF CUPY_HIP_VERSION != 0:
        status = miopenDestroyTensorDescriptor(<TensorDescriptor>filterDesc)
    ELSE:
        status = cudnnDestroyFilterDescriptor(<FilterDescriptor>filterDesc)
    check_status(status)

###############################################################################
# Convolution
###############################################################################

cpdef size_t createConvolutionDescriptor() except? 0:
    cdef ConvolutionDescriptor desc
    IF CUPY_HIP_VERSION != 0:
        status = miopenCreateConvolutionDescriptor(&desc)
    ELSE:
        status = cudnnCreateConvolutionDescriptor(&desc)
    check_status(status)
    return <size_t>desc

cpdef setConvolutionGroupCount(size_t convDesc, int groupCount):
    IF CUPY_HIP_VERSION != 0:
        status = miopenSetConvolutionGroupCount(
            <ConvolutionDescriptor>convDesc, groupCount)
    ELSE:
        status = cudnnSetConvolutionGroupCount(
            <ConvolutionDescriptor>convDesc, groupCount)
    check_status(status)

"""
cpdef int getConvolutionGroupCount(size_t convDesc) except? -1:
    cdef int groupCount
    IF CUPY_HIP_VERSION != 0:
        status = miopenGetConvolutionGroupCount(
            <ConvolutionDescriptor>convDesc, &groupCount)
    ELSE:
        status = cudnnGetConvolutionGroupCount(
            <ConvolutionDescriptor>convDesc, &groupCount)
    check_status(status)
    return groupCount
"""

cpdef setConvolution2dDescriptor_v4(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h,
        int dilation_w, int mode):
    IF CUPY_HIP_VERSION != 0:
        status = miopenInitConvolutionDescriptor(
            <ConvolutionDescriptor>convDesc, <ConvolutionMode>mode, pad_h, pad_w, 
            u, v, dilation_h,dilation_w)
    ELSE:
        status = cudnnSetConvolution2dDescriptor_v4(
            <ConvolutionDescriptor>convDesc, pad_h, pad_w, u, v, dilation_h,
            dilation_w, <ConvolutionMode>mode)
    check_status(status)


cpdef setConvolution2dDescriptor_v5(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h,
        int dilation_w, int mode, size_t computeType):
    IF CUPY_HIP_VERSION != 0:
        status = miopenInitConvolutionDescriptor(
            <ConvolutionDescriptor>convDesc, <ConvolutionMode>mode, pad_h, pad_w,
            u, v, dilation_h,dilation_w)
    ELSE:
        status = cudnnSetConvolution2dDescriptor_v5(
            <ConvolutionDescriptor>convDesc, pad_h, pad_w, u, v, dilation_h,
            dilation_w, <ConvolutionMode>mode, <DataType>computeType)
    check_status(status)


cpdef setConvolutionNdDescriptor_v3(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t dilationA, int mode, int dataType):
    IF CUPY_HIP_VERSION != 0:
        status = miopenInitConvolutionNdDescriptor(
            <ConvolutionDescriptor>convDesc, arrayLength, <int*>padA,
            <int*>filterStrideA, <int*>dilationA, <ConvolutionMode>mode)
    ELSE:
        status = cudnnSetConvolutionNdDescriptor_v3(
            <ConvolutionDescriptor>convDesc, arrayLength, <int*>padA,
            <int*>filterStrideA, <int*>dilationA, <ConvolutionMode>mode,
            <DataType>dataType) 
    check_status(status)


cpdef destroyConvolutionDescriptor(size_t convDesc):
    IF CUPY_HIP_VERSION != 0:
        status = miopenDestroyConvolutionDescriptor(<ConvolutionDescriptor>convDesc)
    ELSE:
        status = cudnnDestroyConvolutionDescriptor(
            <ConvolutionDescriptor>convDesc)
    check_status(status)

cpdef list findConvolutionForwardAlgorithmEx(
        intptr_t handle, size_t xDesc, size_t x, size_t wDesc, size_t w,
        size_t convDesc, size_t yDesc, size_t y, int requestedAlgoCount,
        size_t workSpace, size_t workSpaceSizeInBytes):
    cdef vector.vector[ConvolutionFwdAlgoPerf] perfResults
    cdef int returnedAlgoCount
    perfResults.resize(requestedAlgoCount)
    IF CUPY_HIP_VERSION != 0:
        status = miopenFindConvolutionForwardAlgorithm(
            <Handle> handle, <TensorDescriptor>xDesc, <void*>x,
            <FilterDescriptor>wDesc, <void*>w, <ConvolutionDescriptor>convDesc,
            <TensorDescriptor>yDesc, <void*>y, requestedAlgoCount,
            &returnedAlgoCount, perfResults.data(), <void*>workSpace,
            workSpaceSizeInBytes, True)
    ELSE:
        status = cudnnFindConvolutionForwardAlgorithmEx(
            <Handle> handle, <TensorDescriptor>xDesc, <void*>x,
            <FilterDescriptor>wDesc, <void*>w, <ConvolutionDescriptor>convDesc,
            <TensorDescriptor>yDesc, <void*>y, requestedAlgoCount,
            &returnedAlgoCount, perfResults.data(), <void*>workSpace,
            workSpaceSizeInBytes)
    check_status(status)
    perfResults.resize(returnedAlgoCount)
    return [CuDNNAlgoPerf(p.algo, p.status, p.time, p.memory, -1, -1)
            for p in perfResults]

"""
cpdef list findConvolutionForwardAlgorithmEx_v7(
        intptr_t handle, size_t xDesc, size_t x, size_t wDesc, size_t w,
        size_t convDesc, size_t yDesc, size_t y, int requestedAlgoCount,
        size_t workSpace, size_t workSpaceSizeInBytes):
    cdef vector.vector[ConvolutionFwdAlgoPerf_v7] perfResults
    cdef int returnedAlgoCount
    perfResults.resize(requestedAlgoCount)
    IF CUPY_HIP_VERSION != 0:
        status = miopenFindConvolutionForwardAlgorithm(
            <Handle> handle, <TensorDescriptor>xDesc, <void*>x,
            <FilterDescriptor>wDesc, <void*>w, <ConvolutionDescriptor>convDesc,
            <TensorDescriptor>yDesc, <void*>y, requestedAlgoCount,
            &returnedAlgoCount, perfResults.data(), <void*>workSpace,
            workSpaceSizeInBytes, true)
    ELSE:
        status = cudnnFindConvolutionForwardAlgorithmEx_v7(
            <Handle> handle, <TensorDescriptor>xDesc, <void*>x,
            <FilterDescriptor>wDesc, <void*>w, <ConvolutionDescriptor>convDesc,
            <TensorDescriptor>yDesc, <void*>y, requestedAlgoCount,
            &returnedAlgoCount, perfResults.data(), <void*>workSpace,
            workSpaceSizeInBytes)
    check_status(status)
    perfResults.resize(returnedAlgoCount)
    return [CuDNNAlgoPerf(p.algo, p.status, p.time, p.memory,
                          p.determinism, p.mathType)
            for p in perfResults]
"""

cpdef Py_ssize_t getConvolutionForwardWorkspaceSize(
        intptr_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int algo) except? -1:
    cdef size_t sizeInBytes
    IF CUPY_HIP_VERSION != 0:
        status = miopenConvolutionForwardGetWorkSpaceSize(
            <Handle>handle, <TensorDescriptor>srcDesc,
            <TensorDescriptor>filterDesc, <ConvolutionDescriptor> convDesc,
            <TensorDescriptor>destDesc, &sizeInBytes)
    ELSE:
        status = cudnnGetConvolutionForwardWorkspaceSize(
            <Handle>handle, <TensorDescriptor>srcDesc,
            <FilterDescriptor>filterDesc, <ConvolutionDescriptor> convDesc,
            <TensorDescriptor>destDesc, <ConvolutionFwdAlgo>algo, &sizeInBytes)
    check_status(status)
    return <Py_ssize_t>sizeInBytes


cpdef convolutionForward(
        intptr_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t filterDesc, size_t filterData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t destDesc, size_t destData):
    _setStream(handle)
    with nogil:
        IF CUPY_HIP_VERSION != 0:
            status = miopenConvolutionForward(
                <Handle>handle, <void*>alpha,
                <TensorDescriptor>srcDesc, <void*>srcData,
                <FilterDescriptor>filterDesc, <void*>filterData,
                <ConvolutionDescriptor>convDesc, <ConvolutionFwdAlgo>algo,
                <void*>beta, <TensorDescriptor>destDesc, <void*>destData, 
                <void*>workSpace, workSpaceSizeInBytes)
        ELSE:
            status = cudnnConvolutionForward(
                <Handle>handle, <void*>alpha,
                <TensorDescriptor>srcDesc, <void*>srcData,
                <FilterDescriptor>filterDesc, <void*>filterData,
                <ConvolutionDescriptor>convDesc, <ConvolutionFwdAlgo>algo,
                <void*>workSpace, workSpaceSizeInBytes, <void*>beta,
                <TensorDescriptor>destDesc, <void*>destData)
    check_status(status)


