#include "atomic_fun.cuh"

#define CUDA_CHECKING(x) TORCH_CHECK(x.type().is_cuda(), #x, "must be CUDA Tensor")
#define CONTIGUOUS_CHECKING(x) TORCH_CHECK(x.is_contiguous(), #x, "must be contiguous Tensor")
#define INPUT_CHECKING(x) CUDA_CHECKING(x); CONTIGUOUS_CHECKING(x)

#define THREADS_SCALAR 256
#define THREADS_VEC 128
#define BLOCK_SIZE_RET(N, M) ((N) + (M) - 1) / (M)

enum IdxElemOperateType {Sum, Mul};

const std::map<std::string, IdxElemOperateType> OperateDefine = {
    {"sum", Sum}, {"mul", Mul},
};

#define IDXELEMCUDA(NAME)\
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##Scalar(\
        const scalar_t* src_data1, const scalar_t* src_data2,\
        const int64_t* src_idx1, const int64_t* src_idx2,\
        scalar_t* out_data, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            int64_t offset2 = src_idx2[bid] * D + hid;\
            \
            out_data[tid] = OP(src_data1[offset1], src_data2[offset2]);\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##Scalar(\
        const scalar_t* src_data1, const scalar_t* src_data2,\
        const int64_t* src_idx1,\
        scalar_t* out_data, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            \
            out_data[tid] = OP(src_data1[offset1], src_data2[tid]);\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##Float4(\
        const scalar_t* src_data1, const scalar_t* src_data2,\
        const int64_t* src_idx1, const int64_t* src_idx2,\
        scalar_t* out_data, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        float4 src_cache1 = {};\
        float4 src_cache2 = {};\
        float4 out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            int64_t offset2 = src_idx2[bid] * D + hid;\
            \
            src_cache1 = ((float4 *)src_data1)[offset1];\
            src_cache2 = ((float4 *)src_data2)[offset2];\
            \
            out_cache.x = OP(src_cache1.x, src_cache2.x);\
            out_cache.y = OP(src_cache1.y, src_cache2.y);\
            out_cache.z = OP(src_cache1.z, src_cache2.z);\
            out_cache.w = OP(src_cache1.w, src_cache2.w);\
            \
            ((float4 *)out_data)[tid] = out_cache;\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##Float4(\
        const scalar_t* src_data1, const scalar_t* src_data2,\
        const int64_t* src_idx1,\
        scalar_t* out_data, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        float4 src_cache1 = {};\
        float4 src_cache2 = {};\
        float4 out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            \
            src_cache1 = ((float4 *)src_data1)[offset1];\
            src_cache2 = ((float4 *)src_data2)[tid];\
            \
            out_cache.x = OP(src_cache1.x, src_cache2.x);\
            out_cache.y = OP(src_cache1.y, src_cache2.y);\
            out_cache.z = OP(src_cache1.z, src_cache2.z);\
            out_cache.w = OP(src_cache1.w, src_cache2.w);\
            \
            ((float4 *)out_data)[tid] = out_cache;\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##Half2(\
        const scalar_t* src_data1, const scalar_t* src_data2,\
        const int64_t* src_idx1, const int64_t* src_idx2,\
        scalar_t* out_data, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        __half2 src_cache1 = {};\
        __half2 src_cache2 = {};\
        __half2 out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            int64_t offset2 = src_idx2[bid] * D + hid;\
            \
            src_cache1 = ((__half2*)src_data1)[offset1];\
            src_cache2 = ((__half2*)src_data2)[offset2];\
            \
            out_cache = VECOP(src_cache1, src_cache2);\
            \
            ((__half2*)out_data)[tid] = out_cache;\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##Half2(\
        const scalar_t* src_data1, const scalar_t* src_data2,\
        const int64_t* src_idx1,\
        scalar_t* out_data, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        __half2 src_cache1 = {};\
        __half2 src_cache2 = {};\
        __half2 out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            \
            src_cache1 = ((__half2*)src_data1)[offset1];\
            src_cache2 = ((__half2*)src_data2)[tid];\
            \
            out_cache = VECOP(src_cache1, src_cache2);\
            \
            ((__half2*)out_data)[tid] = out_cache;\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##Float162(\
        const scalar_t* src_data1, const scalar_t* src_data2,\
        const int64_t* src_idx1, const int64_t* src_idx2,\
        scalar_t* out_data, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        __nv_bfloat162 src_cache1 = {};\
        __nv_bfloat162 src_cache2 = {};\
        __nv_bfloat162 out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            int64_t offset2 = src_idx2[bid] * D + hid;\
            \
            src_cache1 = ((__nv_bfloat162*)src_data1)[offset1];\
            src_cache2 = ((__nv_bfloat162*)src_data2)[offset2];\
            \
            out_cache = VECOP(src_cache1, src_cache2);\
            \
            ((__nv_bfloat162*)out_data)[tid] = out_cache;\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##Float162(\
        const scalar_t* src_data1, const scalar_t* src_data2,\
        const int64_t* src_idx1,\
        scalar_t* out_data, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        __nv_bfloat162 src_cache1 = {};\
        __nv_bfloat162 src_cache2 = {};\
        __nv_bfloat162 out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            \
            src_cache1 = ((__nv_bfloat162*)src_data1)[offset1];\
            src_cache2 = ((__nv_bfloat162*)src_data2)[tid];\
            \
            out_cache = VECOP(src_cache1, src_cache2);\
            \
            ((__nv_bfloat162*)out_data)[tid] = out_cache;\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##BackwardScalar(\
        const scalar_t* grad_out, const int64_t* src_idx1,\
        const int64_t* src_idx2, const scalar_t* src_data1,\
        const scalar_t* src_data2, scalar_t* grad_src1,\
        scalar_t* grad_src2, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            int64_t offset2 = src_idx2[bid] * D + hid;\
            \
            scalar_t grad_out_cache = grad_out[tid];\
            \
            atomic_adds(grad_src1 + offset1, OPBWD(grad_out_cache, src_data2[offset2]));\
            atomic_adds(grad_src2 + offset2, OPBWD(grad_out_cache, src_data1[offset1]));\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##BackwardScalar(\
        const scalar_t* grad_out, const int64_t* src_idx1,\
        const scalar_t* src_data1,\
        const scalar_t* src_data2, scalar_t* grad_src1,\
        scalar_t* grad_src2, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            \
            scalar_t grad_out_cache = grad_out[tid];\
            \
            grad_src2[tid] = OPBWD(grad_out_cache, src_data1[offset1]);\
            atomic_adds(grad_src1 + offset1, OPBWD(grad_out_cache, src_data2[tid]));\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##BackwardHalf2(\
        const scalar_t* grad_out, const int64_t* src_idx1,\
        const int64_t* src_idx2, const scalar_t* src_data1,\
        const scalar_t* src_data2, scalar_t* grad_src1,\
        scalar_t* grad_src2, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        __half2 src_cache1 = {};\
        __half2 src_cache2 = {};\
        __half2 grad_out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            int64_t offset2 = src_idx2[bid] * D + hid;\
            \
            src_cache1 = ((__half2*)src_data1)[offset1];\
            src_cache2 = ((__half2*)src_data2)[offset2];\
            grad_out_cache = ((__half2*)grad_out)[tid];\
            \
            atomic_adds(((__half2*)grad_src1) + offset1, VECOPBWD(grad_out_cache, src_cache2));\
            atomic_adds(((__half2*)grad_src2) + offset2, VECOPBWD(grad_out_cache, src_cache1));\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##BackwardHalf2(\
        const scalar_t* grad_out, const int64_t* src_idx1,\
        const scalar_t* src_data1,\
        const scalar_t* src_data2, scalar_t* grad_src1,\
        scalar_t* grad_src2, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        __half2 src_cache1 = {};\
        __half2 src_cache2 = {};\
        __half2 grad_out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            \
            src_cache1 = ((__half2*)src_data1)[offset1];\
            src_cache2 = ((__half2*)src_data2)[tid];\
            grad_out_cache = ((__half2*)grad_out)[tid];\
            \
            ((__half2*)grad_src2)[tid] = VECOPBWD(grad_out_cache, src_cache1);\
            atomic_adds(((__half2*)grad_src1) + offset1, VECOPBWD(grad_out_cache, src_cache2));\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##BackwardFloat162(\
        const scalar_t* grad_out, const int64_t* src_idx1,\
        const int64_t* src_idx2, const scalar_t* src_data1,\
        const scalar_t* src_data2, scalar_t* grad_src1,\
        scalar_t* grad_src2, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        __nv_bfloat162 src_cache1 = {};\
        __nv_bfloat162 src_cache2 = {};\
        __nv_bfloat162 grad_out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            int64_t offset2 = src_idx2[bid] * D + hid;\
            \
            src_cache1 = ((__nv_bfloat162*)src_data1)[offset1];\
            src_cache2 = ((__nv_bfloat162*)src_data2)[offset2];\
            grad_out_cache = ((__nv_bfloat162*)grad_out)[tid];\
            \
            atomic_adds(((__nv_bfloat162*)grad_src1) + offset1, VECOPBWD(grad_out_cache, src_cache2));\
            atomic_adds(((__nv_bfloat162*)grad_src2) + offset2, VECOPBWD(grad_out_cache, src_cache1));\
        }\
    }\
    \
    template <typename scalar_t>\
    __global__ void IndexElemwise##NAME##BackwardFloat162(\
        const scalar_t* grad_out, const int64_t* src_idx1,\
        const scalar_t* src_data1,\
        const scalar_t* src_data2, scalar_t* grad_src1,\
        scalar_t* grad_src2, int D, int numel\
    ) {\
        int tid = blockIdx.x * blockDim.x + threadIdx.x;\
        int bid = tid / D;\
        int hid = tid % D;\
        \
        __nv_bfloat162 src_cache1 = {};\
        __nv_bfloat162 src_cache2 = {};\
        __nv_bfloat162 grad_out_cache = {};\
        \
        if (tid < numel) {\
            int64_t offset1 = src_idx1[bid] * D + hid;\
            \
            src_cache1 = ((__nv_bfloat162*)src_data1)[offset1];\
            src_cache2 = ((__nv_bfloat162*)src_data2)[tid];\
            grad_out_cache = ((__nv_bfloat162*)grad_out)[tid];\
            \
            ((__nv_bfloat162*)grad_src2)[tid] = VECOPBWD(grad_out_cache, src_cache1);\
            atomic_adds(((__nv_bfloat162*)grad_src1) + offset1, VECOPBWD(grad_out_cache, src_cache2));\
        }\
    }\


#define fwd_args(scalar_t)\
    scalar_t* src_data1, scalar_t* src_data2,\
    int64_t* src_idx1, int64_t* src_idx2,\
    scalar_t* out_data, int D, int numel, bool vectorize\

#define fwd_in\
    src_data1, src_data2, src_idx1, src_idx2, out_data\

#define fwd_in_no_idx2\
    src_data1, src_data2, src_idx1, out_data\

#define bwd_args(scalar_t)\
    scalar_t* grad_out, int64_t* src_idx1,\
    int64_t* src_idx2, scalar_t* src_data1,\
    scalar_t* src_data2, scalar_t* grad_src1,\
    scalar_t* grad_src2, int D, int numel, bool vectorize\

#define bwd_in\
    grad_out, src_idx1, src_idx2, src_data1, src_data2, grad_src1, grad_src2\

#define bwd_in_no_idx2\
    grad_out, src_idx1, src_data1, src_data2, grad_src1, grad_src2\


#define OP(X, Y) X + Y
#define VECOP(X, Y) __hadd2(X, Y)
#define OPBWD(X, Y) X
#define VECOPBWD(X, Y) X
IDXELEMCUDA(Sum)
#undef OP
#undef VECOP
#undef OPBWD
#undef VECOPBWD
void idxSum(fwd_args(uint8_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumScalar<uint8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseSumScalar<uint8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxSum(fwd_args(int8_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumScalar<int8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseSumScalar<int8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxSum(fwd_args(int16_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumScalar<int16_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseSumScalar<int16_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxSum(fwd_args(int)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumScalar<int><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseSumScalar<int><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxSum(fwd_args(int64_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumScalar<int64_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseSumScalar<int64_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxSum(fwd_args(double)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumScalar<double><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseSumScalar<double><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxSum(fwd_args(float)) {
    if (vectorize) {
        if (src_idx2 != nullptr)
            IndexElemwiseSumFloat4<float><<<BLOCK_SIZE_RET(numel >> 2, THREADS_VEC), THREADS_VEC>>>(
                fwd_in, D >> 2, numel >> 2
            );
        else
            IndexElemwiseSumFloat4<float><<<BLOCK_SIZE_RET(numel >> 2, THREADS_VEC), THREADS_VEC>>>(
                fwd_in_no_idx2, D >> 2, numel >> 2
            );
    }
    else {
        if (src_idx2 != nullptr)
            IndexElemwiseSumScalar<float><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in, D, numel
            );
        else
            IndexElemwiseSumScalar<float><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in_no_idx2, D, numel
            );
    }
}
void idxSum(fwd_args(c10::Half)) {
    if (vectorize) {
        if (src_idx2 != nullptr)
            IndexElemwiseSumHalf2<c10::Half><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(
                fwd_in, D >> 1, numel >> 1
            );
        else
            IndexElemwiseSumHalf2<c10::Half><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(
                fwd_in_no_idx2, D >> 1, numel >> 1
            );
    }
    else {
        if (src_idx2 != nullptr)
            IndexElemwiseSumScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in, D, numel
            );
        else
            IndexElemwiseSumScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in_no_idx2, D, numel
            );
    }
}
void idxSum(fwd_args(c10::BFloat16)) {
    if (vectorize) {
        if (src_idx2 != nullptr)
            IndexElemwiseSumFloat162<c10::BFloat16><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(
                fwd_in, D >> 1, numel >> 1
            );
        else
            IndexElemwiseSumFloat162<c10::BFloat16><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(
                fwd_in_no_idx2, D >> 1, numel >> 1
            );
    }
    else {
        if (src_idx2 != nullptr)
            IndexElemwiseSumScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in, D, numel
            );
        else
            IndexElemwiseSumScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in_no_idx2, D, numel
            );
    }
}

void idxSumBwd(bwd_args(uint8_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumBackwardScalar<uint8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseSumBackwardScalar<uint8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxSumBwd(bwd_args(int8_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumBackwardScalar<int8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseSumBackwardScalar<int8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxSumBwd(bwd_args(int16_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumBackwardScalar<int16_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseSumBackwardScalar<int16_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxSumBwd(bwd_args(int)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumBackwardScalar<int><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseSumBackwardScalar<int><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxSumBwd(bwd_args(int64_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumBackwardScalar<int64_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseSumBackwardScalar<int64_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxSumBwd(bwd_args(double)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumBackwardScalar<double><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseSumBackwardScalar<double><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxSumBwd(bwd_args(float)) {
    if (src_idx2 != nullptr)
        IndexElemwiseSumBackwardScalar<float><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseSumBackwardScalar<float><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
    void idxSumBwd(bwd_args(c10::Half)) {
        if (src_idx2 != nullptr)
            IndexElemwiseSumBackwardScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
        else
            IndexElemwiseSumBackwardScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
    }
#else
    void idxSumBwd(bwd_args(c10::Half)) {
        if (vectorize) {
            if (src_idx2 != nullptr)
                IndexElemwiseSumBackwardHalf2<c10::Half><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(bwd_in, D >> 1, numel >> 1);
            else
                IndexElemwiseSumBackwardHalf2<c10::Half><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(bwd_in_no_idx2, D >> 1, numel >> 1);
        }
        else {
            if (src_idx2 != nullptr)
                IndexElemwiseSumBackwardScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
            else
                IndexElemwiseSumBackwardScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
        }
    }
#endif

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
    void idxSumBwd(bwd_args(c10::BFloat16)) {
        if (src_idx2 != nullptr)
            IndexElemwiseSumBackwardScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
        else
            IndexElemwiseSumBackwardScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
    }
#else
    void idxSumBwd(bwd_args(c10::BFloat16)) {
        if (vectorize) {
            if (src_idx2 != nullptr)
                IndexElemwiseSumBackwardFloat162<c10::BFloat16><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(bwd_in, D >> 1, numel >> 1);
            else
                IndexElemwiseSumBackwardFloat162<c10::BFloat16><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(bwd_in_no_idx2, D >> 1, numel >> 1);
        }
        else {
            if (src_idx2 != nullptr)
                IndexElemwiseSumBackwardScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
            else
                IndexElemwiseSumBackwardScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
        }
    }
#endif

#define OP(X, Y) X * Y
#define VECOP(X, Y) __hmul2(X, Y)
#define OPBWD(X, Y) X * Y
#define VECOPBWD(X, Y) __hmul2(X, Y)
IDXELEMCUDA(Mul)
#undef OP
#undef VECOP
#undef OPBWD
#undef VECOPBWD
void idxMul(fwd_args(uint8_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulScalar<uint8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseMulScalar<uint8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxMul(fwd_args(int8_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulScalar<int8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseMulScalar<int8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxMul(fwd_args(int16_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulScalar<int16_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseMulScalar<int16_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxMul(fwd_args(int)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulScalar<int><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseMulScalar<int><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxMul(fwd_args(int64_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulScalar<int64_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseMulScalar<int64_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxMul(fwd_args(double)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulScalar<double><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in, D, numel
        );
    else
        IndexElemwiseMulScalar<double><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
            fwd_in_no_idx2, D, numel
        );
}
void idxMul(fwd_args(float)) {
    if (vectorize) {
        if (src_idx2 != nullptr)
            IndexElemwiseMulFloat4<float><<<BLOCK_SIZE_RET(numel >> 2, THREADS_VEC), THREADS_VEC>>>(
                fwd_in, D >> 2, numel >> 2
            );
        else
            IndexElemwiseMulFloat4<float><<<BLOCK_SIZE_RET(numel >> 2, THREADS_VEC), THREADS_VEC>>>(
                fwd_in_no_idx2, D >> 2, numel >> 2
            );
    }
    else {
        if (src_idx2 != nullptr)
            IndexElemwiseMulScalar<float><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in, D, numel
            );
        else
            IndexElemwiseMulScalar<float><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in_no_idx2, D, numel
            );
    }
}
void idxMul(fwd_args(c10::Half)) {
    if (vectorize) {
        if (src_idx2 != nullptr)
            IndexElemwiseMulHalf2<c10::Half><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(
                fwd_in, D >> 1, numel >> 1
            );
        else
            IndexElemwiseMulHalf2<c10::Half><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(
                fwd_in_no_idx2, D >> 1, numel >> 1
            );
    }
    else {
        if (src_idx2 != nullptr)
            IndexElemwiseMulScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in, D, numel
            );
        else
            IndexElemwiseMulScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in_no_idx2, D, numel
            );
    }
}
void idxMul(fwd_args(c10::BFloat16)) {
    if (vectorize) {
        if (src_idx2 != nullptr)
            IndexElemwiseMulFloat162<c10::BFloat16><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(
                fwd_in, D >> 1, numel >> 1
            );
        else
            IndexElemwiseMulFloat162<c10::BFloat16><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(
                fwd_in_no_idx2, D >> 1, numel >> 1
            );
    }
    else {
        if (src_idx2 != nullptr)
            IndexElemwiseMulScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in, D, numel
            );
        else
            IndexElemwiseMulScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(
                fwd_in_no_idx2, D, numel
            );
    }
}

void idxMulBwd(bwd_args(uint8_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulBackwardScalar<uint8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseMulBackwardScalar<uint8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxMulBwd(bwd_args(int8_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulBackwardScalar<int8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseMulBackwardScalar<int8_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxMulBwd(bwd_args(int16_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulBackwardScalar<int16_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseMulBackwardScalar<int16_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxMulBwd(bwd_args(int)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulBackwardScalar<int><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseMulBackwardScalar<int><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxMulBwd(bwd_args(int64_t)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulBackwardScalar<int64_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseMulBackwardScalar<int64_t><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxMulBwd(bwd_args(double)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulBackwardScalar<double><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseMulBackwardScalar<double><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}
void idxMulBwd(bwd_args(float)) {
    if (src_idx2 != nullptr)
        IndexElemwiseMulBackwardScalar<float><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
    else
        IndexElemwiseMulBackwardScalar<float><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
}

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
    void idxMulBwd(bwd_args(c10::Half)) {
        if (src_idx2 != nullptr)
            IndexElemwiseMulBackwardScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
        else
            IndexElemwiseMulBackwardScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
    }
#else
    void idxMulBwd(bwd_args(c10::Half)) {
        if (vectorize) {
            if (src_idx2 != nullptr)
                IndexElemwiseMulBackwardHalf2<c10::Half><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(bwd_in, D >> 1, numel >> 1);
            else
                IndexElemwiseMulBackwardHalf2<c10::Half><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(bwd_in_no_idx2, D >> 1, numel >> 1);
        }
        else {
            if (src_idx2 != nullptr)
                IndexElemwiseMulBackwardScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
            else
                IndexElemwiseMulBackwardScalar<c10::Half><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
        }
    }
#endif

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
    void idxMulBwd(bwd_args(c10::BFloat16)) {
        if (src_idx2 != nullptr)
            IndexElemwiseMulBackwardScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
        else
            IndexElemwiseMulBackwardScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
    }
#else
    void idxMulBwd(bwd_args(c10::BFloat16)) {
        if (vectorize) {
            if (src_idx2 != nullptr)
                IndexElemwiseMulBackwardFloat162<c10::BFloat16><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(bwd_in, D >> 1, numel >> 1);
            else
                IndexElemwiseMulBackwardFloat162<c10::BFloat16><<<BLOCK_SIZE_RET(numel >> 1, THREADS_VEC), THREADS_VEC>>>(bwd_in_no_idx2, D >> 1, numel >> 1);
        }
        else {
            if (src_idx2 != nullptr)
                IndexElemwiseMulBackwardScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in, D, numel);
            else
                IndexElemwiseMulBackwardScalar<c10::BFloat16><<<BLOCK_SIZE_RET(numel, THREADS_SCALAR), THREADS_SCALAR>>>(bwd_in_no_idx2, D, numel);
        }
    }
#endif

#undef fwd_args
#undef fwd_in
#undef fwd_in_no_idx2
#undef bwd_args
#undef bwd_in
#undef bwd_in_no_idx2