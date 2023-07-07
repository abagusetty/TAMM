#pragma once

#include "tamm/gpu_streams.hpp"

namespace tamm::kernels {

namespace cpu {
template<typename T, typename T1, typename T2, typename T3>
void blas(int m, int n, int k, const T alpha, const T2* A, int lda, const T3* B, int ldb,
          const T beta, T1* C, int ldc);
} // namespace cpu

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)  
namespace gpu {
template<typename T, typename T1, typename T2, typename T3>
void blas(int n, int m, int k, const T alpha, const T3* B, int ldb, const T2* A, int lda,
          const T beta, T1* C, int ldc, gpuStream_t& blashandle);

template<typename T>
void transpose_inplace(T* out, const T* in, int* outDims, int* inDims, int* rdims, const bool conjugate,
                       gpuStream_t& thandle);
#endif
  
} // namespace gpu

} // namespace tamm::kernels
