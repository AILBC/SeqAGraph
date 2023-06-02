#pragma once

#define ATOMICFUN(NAME)\
    template<typename scalar, size_t size> struct Atomic##NAME##IntImpl;\
    \
    template<typename scalar> struct Atomic##NAME##IntImpl<scalar, 1> {\
        static inline __device__ void apply(scalar* addr, scalar val) {\
            uint32_t* addr_as_ui = (uint32_t*)(addr - ((size_t)addr & 0b11));\
            uint32_t old = *addr_as_ui;\
            uint32_t shift = ((size_t)addr & 0b11) << 3;\
            uint32_t sum;\
            uint32_t assumed;\
            \
            do {\
                assumed = old;\
                sum = ATOMOP(val, (scalar)((old >> shift) & 0xff));\
                old = (old & ~(0x000000ff << shift)) | (sum << shift);\
                old = atomicCAS(addr_as_ui, assumed, old);\
            } while (assumed != old);\
        }\
    };\
    \
    template<typename scalar> struct Atomic##NAME##IntImpl<scalar, 2> {\
        static inline __device__ void apply(scalar* addr, scalar val) {\
            uint32_t* addr_as_ui = (uint32_t*)((char*)addr - ((size_t)addr & 0b10));\
            uint32_t old = *addr_as_ui;\
            uint32_t sum;\
            uint32_t assumed;\
            uint32_t newval;\
            \
            do {\
                assumed = old;\
                sum = ATOMOP(val, (size_t)addr & 0b10 ? (scalar)(old >> 16) : (scalar)(old & 0xffff));\
                newval = (size_t)addr & 0b10 ? (old & 0xffff) | (sum << 16) : (old & 0xffff0000) | sum;\
                old = atomicCAS(addr_as_ui, assumed, newval);\
            } while (assumed != old);\
        }\
    };\
    \
    template<typename scalar> struct Atomic##NAME##IntImpl<scalar, 4> {\
        static inline __device__ void apply(scalar* addr, scalar val) {\
            uint32_t* addr_as_ui = (uint32_t*)addr;\
            uint32_t old = *addr_as_ui;\
            uint32_t assumed;\
            \
            do {\
                assumed = old;\
                old = atomicCAS(addr_as_ui, assumed, ATOMOP(val, (scalar)old));\
            } while (assumed != old);\
        }\
    };\
    \
    template<typename scalar> struct Atomic##NAME##IntImpl<scalar, 8> {\
        static inline __device__ void apply(scalar* addr, scalar val) {\
            unsigned long long int* addr_as_ui = (unsigned long long int*)addr;\
            unsigned long long int old = *addr_as_ui;\
            unsigned long long int assumed;\
            \
            do {\
                assumed = old;\
                old = atomicCAS(addr_as_ui, assumed, ATOMOP(val, (scalar)old));\
            } while (assumed != old);\
        }\
    };\
    \
    template<typename scalar, size_t size> struct Atomic##NAME##FloatImpl;\
    \
    template<> struct Atomic##NAME##FloatImpl<c10::Half, 2> {\
        static inline __device__ void apply(c10::Half* addr, c10::Half val) {\
            unsigned int* addr_as_ui = (unsigned int*)((char*)addr - ((size_t)addr & 0b10));\
            unsigned int old = *addr_as_ui;\
            unsigned int assumed;\
            \
            do {\
                assumed = old;\
                c10::Half hsum;\
                hsum.x = (size_t)addr & 0b10 ? (old >> 16) : (old & 0xffff);\
                hsum = ATOMOP(hsum, val);\
                old = (size_t)addr & 0b10 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;\
                old = atomicCAS(addr_as_ui, assumed, old);\
            } while (assumed != old);\
        }\
    };\
    \
    template<> struct Atomic##NAME##FloatImpl<c10::BFloat16, 2> {\
        static inline __device__ void apply(c10::BFloat16* addr, c10::BFloat16 val) {\
            unsigned int* addr_as_ui = (unsigned int*)((char*)addr - ((size_t)addr & 0b10));\
            unsigned int old = *addr_as_ui;\
            unsigned int assumed;\
            \
            do {\
                assumed = old;\
                c10::BFloat16 bsum;\
                bsum.x = (size_t)addr & 0b10 ? (old >> 16) : (old & 0xffff);\
                bsum = ATOMOP(bsum, val);\
                old = (size_t)addr & 0b10 ? (old & 0xffff) | (bsum.x << 16) : (old & 0xffff0000) | bsum.x;\
                old = atomicCAS(addr_as_ui, assumed, old);\
            } while (assumed != old);\
        }\
    };\
    \
    template<typename scalar> struct Atomic##NAME##FloatImpl<scalar, 4> {\
        static inline __device__ void apply(scalar* addr, scalar val) {\
            int* addr_as_ui = (int*)addr;\
            int old = *addr_as_ui;\
            int assumed;\
            \
            do {\
                assumed = old;\
                old = atomicCAS(addr_as_ui, assumed, __float_as_int(ATOMOP(val, __int_as_float(assumed))));\
            } while (assumed != old);\
        }\
    };\
    \
    template<typename scalar> struct Atomic##NAME##FloatImpl<scalar, 8> {\
        static inline __device__ void apply(scalar* addr, scalar val) {\
            unsigned long long int* addr_as_ui = (unsigned long long int*)addr;\
            unsigned long long int old = *addr_as_ui;\
            unsigned long long int assumed;\
            \
            do {\
                assumed = old;\
                old = atomicCAS(addr_as_ui, assumed, __double_as_longlong(ATOMOP(val, __longlong_as_double(assumed))));\
            } while (assumed != old);\
        }\
    };\

#define ATOMOP(X, Y) (Y) + (X)
ATOMICFUN(Add)
#undef ATOMOP

#define atomic_args(dtype) dtype* addr, dtype val
#define atomic_in addr, val

static inline __device__ void atomic_adds(atomic_args(uint8_t)) {
    AtomicAddIntImpl<uint8_t, sizeof(uint8_t)>::apply(atomic_in);
}
static inline __device__ void atomic_adds(atomic_args(int8_t)) {
    AtomicAddIntImpl<int8_t, sizeof(int8_t)>::apply(atomic_in);
}
static inline __device__ void atomic_adds(atomic_args(int16_t)) {
    AtomicAddIntImpl<int16_t, sizeof(int16_t)>::apply(atomic_in);
}
static inline __device__ void atomic_adds(atomic_args(int32_t)) {
    atomicAdd(atomic_in);
}
static inline __device__ void atomic_adds(atomic_args(int64_t)) {
    AtomicAddIntImpl<int64_t, sizeof(int64_t)>::apply(atomic_in);
}
static inline __device__ void atomic_adds(atomic_args(float)) {
    atomicAdd(atomic_in);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
    static inline __device__ void atomic_adds(atomic_args(double)) {
        AtomicAddFloatImpl<double, sizeof(double)>::apply(atomic_in);
    }
#else
    static inline __device__ void atomic_adds(atomic_args(double)) {
        atomicAdd(atomic_in);
    }
#endif

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
    static inline __device__ void atomic_adds(atomic_args(c10::Half)) {
        AtomicAddFloatImpl<c10::Half, sizeof(c10::Half)>::apply(atomic_in);
    }
#else
    static inline __device__ void atomic_adds(atomic_args(c10::Half)) {
        atomicAdd(reinterpret_cast<__half*>(addr), val);
    }
    static inline __device__ void atomic_adds(atomic_args(__half2)) {
        atomicAdd(atomic_in);
    }
#endif

#if defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)))
    static inline __device__ void atomic_adds(atomic_args(c10::BFloat16)) {
        AtomicAddFloatImpl<c10::BFloat16, sizeof(c10::BFloat16)>::apply(atomic_in);
    }
#else
    static inline __device__ void atomic_adds(atomic_args(c10::BFloat16)) {
        atomicAdd(reinterpret_cast<__nv_bfloat16*>(addr), *reinterpret_cast<__nv_bfloat16*>(&val));
    }
    static inline __device__ void atomic_adds(atomic_args(__nv_bfloat162)) {
        atomicAdd(atomic_in);
    }
#endif

#undef atomic_args
#undef atomic_in