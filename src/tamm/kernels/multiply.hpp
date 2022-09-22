#pragma once

#include "tamm/errors.hpp"
#include "tamm/kernels/assign.hpp"
#include "tamm/types.hpp"

#include <algorithm>
#include <complex>
#include <numeric>
#include <vector>

#include "ga/ga_linalg.h"

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#include "librett/librett.h"
#include "tamm/gpu_memory_pool.hpp"
#else
namespace tamm {
using gpuStream_t = int; // not used
}
#endif

#if defined(USE_CUDA)

#define CUBLAS_CHECK(err)                                                              \
  do {                                                                                 \
    cublasStatus_t err_ = (err);                                                       \
    if(err_ != CUBLAS_STATUS_SUCCESS) {                                                \
      std::printf("CUBLAS Exception code: %d at %s : %d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                                        \
    }                                                                                  \
  } while(0)

#define CUDA_CHECK(err)                                                                       \
  do {                                                                                        \
    cudaError_t err_ = (err);                                                                 \
    if(err_ != cudaSuccess) {                                                                 \
      std::printf("CUDA Exception code: %s at %s : %d\n", cudaGetErrorString(err_), __FILE__, \
                  __LINE__);                                                                  \
      throw std::runtime_error("cuda runtime error");                                         \
    }                                                                                         \
  } while(0)
#endif // USE_CUDA

#if defined(USE_HIP)

#define ROCBLAS_CHECK(err)                                                            \
  do {                                                                                \
    rocblas_status err_ = (err);                                                      \
    if(err_ != rocblas_status_success) {                                              \
      std::printf("ROCBLAS Exception code: %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("rocblas error");                                      \
    }                                                                                 \
  } while(0)

#define HIP_CHECK(err)                                                                      \
  do {                                                                                      \
    hipError_t err_ = (err);                                                                \
    if(err_ != hipSuccess) {                                                                \
      std::printf("HIP Exception code: %s at %s : %d\n", hipGetErrorString(err_), __FILE__, \
                  __LINE__);                                                                \
      throw std::runtime_error("hip runtime error");                                        \
    }                                                                                       \
  } while(0)
#endif // USE_HIP

#if defined(USE_DPCPP)
#include "oneapi/mkl.hpp"
#endif // USE_DPCPP

namespace tamm {

namespace kernels {

template<typename T1>
void stream_synchronize(gpuStream_t& shandle) {
#ifdef USE_CUDA
  cudaStreamSynchronize(shandle);
#elif defined(USE_HIP)
  hipStreamSynchronize(shandle);
#elif defined(USE_DPCPP)
  shandle.wait();
#endif
}

template<typename T2, typename T3>
void copy_data_to_gpu_trans(bool& isgpuOp, gpuStream_t& thandle, const T2* ainter_buf, size_t asize,
                            T2** ainter_buf_dev, const T3* binter_buf, size_t bsize,
                            T3** binter_buf_dev) {
  if(!isgpuOp) return;

#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))

    // host-->device copy
#if defined(USE_DPCPP)
  thandle.memcpy(*ainter_buf_dev, ainter_buf, asize * sizeof(T2));
  thandle.memcpy(*binter_buf_dev, binter_buf, bsize * sizeof(T3));
#elif defined(USE_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(*ainter_buf_dev, ainter_buf, asize * sizeof(T2),
                             cudaMemcpyHostToDevice, thandle));
  CUDA_CHECK(cudaMemcpyAsync(*binter_buf_dev, binter_buf, bsize * sizeof(T3),
                             cudaMemcpyHostToDevice, thandle));
#elif defined(USE_HIP)
  HIP_CHECK(hipMemcpyAsync(*ainter_buf_dev, ainter_buf, asize * sizeof(T2), hipMemcpyHostToDevice,
                           thandle));
  HIP_CHECK(hipMemcpyAsync(*binter_buf_dev, binter_buf, bsize * sizeof(T3), hipMemcpyHostToDevice,
                           thandle));
#endif

#endif
}

template<typename T2, typename T3>
void copy_data_to_gpu(bool& isgpuOp, gpuStream_t& thandle, const std::vector<T2>& ainter_buf,
                      T2** ainter_buf_dev, const std::vector<T3>& binter_buf, T3** binter_buf_dev) {
  copy_data_to_gpu_trans(isgpuOp, thandle, ainter_buf.data(), ainter_buf.size(), ainter_buf_dev,
                         binter_buf.data(), binter_buf.size(), binter_buf_dev);
}

template<typename T, typename T1, typename T2, typename T3>
void gemm_wrapper(bool& isgpuOp, gpuStream_t& thandle, int AR, int BR, int B, int M, int N, int K,
                  T alpha, T beta, std::vector<T2>& ainter_buf, const T2* ainter_buf_dev,
                  std::vector<T3>& binter_buf, const T3* binter_buf_dev,
                  std::vector<T1>& cinter_buf, T1* cinter_buf_dev) {
#if defined(USE_CUDA) || defined(USE_HIP)
  auto& handle = tamm::GPUStreamPool::getInstance().getBlasHandle();
#endif

  int ainter_ld  = K;
  int binter_ld  = N;
  int cinter_ld  = N;
  int cbatch_ld  = M * N;
  int abatch_ld  = M * K;
  int bbatch_ld  = K * N;
  int areduce_ld = B * abatch_ld;
  int breduce_ld = B * bbatch_ld;

  for(size_t ari = 0; ari < AR; ari++) {
    for(size_t bri = 0; bri < BR; bri++) {
      for(size_t i = 0; i < B; i++) {
#if defined(USE_DPCPP)
        if(isgpuOp) {
          try {
            if constexpr(!std::is_floating_point_v<T>) {
              if(N == 1 && M == 1 && alpha.real() == 1.0 && beta.real() == 0.0) {
                auto ddotu = oneapi::mkl::blas::column_major::dotu(
                  thandle, K, binter_buf_dev + bri * breduce_ld + i * bbatch_ld, 1,
                  ainter_buf_dev + ari * areduce_ld + i * abatch_ld, 1,
                  cinter_buf_dev + i * cbatch_ld);
                ddotu.wait();
              }
            }
            else {
              if(N == 1 && M == 1 && alpha == 1.0 && beta == 0.0) {
                auto ddot = oneapi::mkl::blas::column_major::dot(
                  thandle, K, binter_buf_dev + bri * breduce_ld + i * bbatch_ld, 1,
                  ainter_buf_dev + ari * areduce_ld + i * abatch_ld, 1,
                  cinter_buf_dev + i * cbatch_ld);
                ddot.wait();
              }
            }

            if(N == 1 && M != 1 && K != 1) {
              auto dgemv = oneapi::mkl::blas::column_major::gemv(
                thandle, oneapi::mkl::transpose::T, K, M, alpha,
                ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld,
                binter_buf_dev + bri * breduce_ld + i * bbatch_ld, 1, beta,
                cinter_buf_dev + i * cbatch_ld, 1);
              dgemv.wait();
            }
            else {
              auto dgemm = oneapi::mkl::blas::column_major::gemm(
                thandle, oneapi::mkl::transpose::N, oneapi::mkl::transpose::N, N, M, K, alpha,
                binter_buf_dev + bri * breduce_ld + i * bbatch_ld, binter_ld,
                ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld, beta,
                cinter_buf_dev + i * cbatch_ld, cinter_ld);
              dgemm.wait();
            }
          } catch(oneapi::mkl::exception const& ex) {
            std::stringstream msg;
            msg << "oneMKL Exception at " << __FILE__ << " : " << __LINE__ << std::endl;
            throw(std::runtime_error(ex.what()));
          }

          continue;
        }
#elif defined(USE_CUDA)
        if(isgpuOp) {
          if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
                       internal::is_complex_v<T3>) {
            if(N == 1 && M == 1 && alpha.real() == 1.0 && beta.real() == 0.0) {
              CUBLAS_CHECK(cublasZdotu(
                handle, K,
                (const cuDoubleComplex*) binter_buf_dev + bri * breduce_ld + i * bbatch_ld, 1,
                (const cuDoubleComplex*) ainter_buf_dev + ari * areduce_ld + i * abatch_ld, 1,
                (cuDoubleComplex*) cinter_buf_dev + i * cbatch_ld));
            }
            else if(N == 1 && M != 1 && K != 1) {
              CUBLAS_CHECK(cublasZgemv(
                handle, CUBLAS_OP_T, K, M, (const cuDoubleComplex*) &alpha,
                (const cuDoubleComplex*) ainter_buf_dev + ari * areduce_ld + i * abatch_ld,
                ainter_ld,
                (const cuDoubleComplex*) binter_buf_dev + bri * breduce_ld + i * bbatch_ld, 1,
                (const cuDoubleComplex*) &beta, (cuDoubleComplex*) cinter_buf_dev + i * cbatch_ld,
                1));
            }
            else {
              CUBLAS_CHECK(cublasZgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, (const cuDoubleComplex*) &alpha,
                (const cuDoubleComplex*) binter_buf_dev + bri * breduce_ld + i * bbatch_ld,
                binter_ld,
                (const cuDoubleComplex*) ainter_buf_dev + ari * areduce_ld + i * abatch_ld,
                ainter_ld, (const cuDoubleComplex*) &beta,
                (cuDoubleComplex*) cinter_buf_dev + i * cbatch_ld, cinter_ld));
            }
          }
          else {
            if(N == 1 and M == 1 and alpha == 1.0 and beta == 0.0) {
              CUBLAS_CHECK(cublasDdot(handle, K, binter_buf_dev + bri * breduce_ld + i * bbatch_ld,
                                      1, ainter_buf_dev + ari * areduce_ld + i * abatch_ld, 1,
                                      cinter_buf_dev + i * cbatch_ld));
            }
            else if(N == 1 and M != 1 and K != 1) {
              CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_T, K, M, &alpha,
                                       ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld,
                                       binter_buf_dev + bri * breduce_ld + i * bbatch_ld, 1, &beta,
                                       cinter_buf_dev + i * cbatch_ld, 1));
            }
            else {
              CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                       binter_buf_dev + bri * breduce_ld + i * bbatch_ld, binter_ld,
                                       ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld,
                                       &beta, cinter_buf_dev + i * cbatch_ld, cinter_ld));
            }
          }

          continue;
        }
#elif defined(USE_HIP)
        if(isgpuOp) {
          if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
                       internal::is_complex_v<T3>) {
            if(N == 1 && M == 1 && alpha.real() == 1.0 && beta.real() == 0.0) {
              ROCBLAS_CHECK(rocblas_zdotu(
                handle, K,
                (const rocblas_double_complex*) binter_buf_dev + bri * breduce_ld + i * bbatch_ld,
                1,
                (const rocblas_double_complex*) ainter_buf_dev + ari * areduce_ld + i * abatch_ld,
                1, (rocblas_double_complex*) cinter_buf_dev + i * cbatch_ld));
            }
            else if(N == 1 and M != 1 and K != 1) {
              ROCBLAS_CHECK(rocblas_zgemv(
                handle, rocblas_operation_transpose, K, M, (const rocblas_double_complex*) &alpha,
                (const rocblas_double_complex*) ainter_buf_dev + ari * areduce_ld + i * abatch_ld,
                ainter_ld,
                (const rocblas_double_complex*) binter_buf_dev + bri * breduce_ld + i * bbatch_ld,
                1, (const rocblas_double_complex*) &beta,
                (rocblas_double_complex*) cinter_buf_dev + i * cbatch_ld, 1));
            }
            else {
              ROCBLAS_CHECK(rocblas_zgemm(
                handle, rocblas_operation_none, rocblas_operation_none, N, M, K,
                (const rocblas_double_complex*) &alpha,
                (const rocblas_double_complex*) binter_buf_dev + bri * breduce_ld + i * bbatch_ld,
                binter_ld,
                (const rocblas_double_complex*) ainter_buf_dev + ari * areduce_ld + i * abatch_ld,
                ainter_ld, (const rocblas_double_complex*) &beta,
                (rocblas_double_complex*) cinter_buf_dev + i * cbatch_ld, cinter_ld));
            }
          }
          else {
            if(N == 1 and M == 1 and alpha == 1.0 and beta == 0.0) {
              ROCBLAS_CHECK(rocblas_ddot(handle, K,
                                         binter_buf_dev + bri * breduce_ld + i * bbatch_ld, 1,
                                         ainter_buf_dev + ari * areduce_ld + i * abatch_ld, 1,
                                         cinter_buf_dev + i * cbatch_ld));
            }
            else if(N == 1 and M != 1 and K != 1) {
              ROCBLAS_CHECK(rocblas_dgemv(handle, rocblas_operation_transpose, K, M, &alpha,
                                          ainter_buf_dev + ari * areduce_ld + i * abatch_ld,
                                          ainter_ld,
                                          binter_buf_dev + bri * breduce_ld + i * bbatch_ld, 1,
                                          &beta, cinter_buf_dev + i * cbatch_ld, 1));
            }
            else {
              ROCBLAS_CHECK(
                rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none, N, M, K,
                              &alpha, binter_buf_dev + bri * breduce_ld + i * bbatch_ld, binter_ld,
                              ainter_buf_dev + ari * areduce_ld + i * abatch_ld, ainter_ld, &beta,
                              cinter_buf_dev + i * cbatch_ld, cinter_ld));
            }
          }
          
          continue;
        }
#endif
        // CPU only ops (i.e., for CPU builds or isgpuOp=false)
        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, M, N, K, alpha,
                    ainter_buf.data() + ari * areduce_ld + i * abatch_ld, ainter_ld,
                    binter_buf.data() + bri * breduce_ld + i * bbatch_ld, binter_ld, beta,
                    cinter_buf.data() + i * cbatch_ld, cinter_ld);

      } // for-i
    }   // for-bri
  }     // for-ari
}

template<typename T1>
void copy_result_to_host(ExecutionHW hw, gpuStream_t& thandle, std::vector<T1>& cinter_buf,
                         T1* cinter_buf_dev) {
  if(hw != ExecutionHW::GPU) return;

// device-->host copy
#if defined(USE_DPCPP)
  thandle.memcpy(cinter_buf.data(), cinter_buf_dev, cinter_buf.size() * sizeof(T1));
#elif defined(USE_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(cinter_buf.data(), cinter_buf_dev, cinter_buf.size() * sizeof(T1),
                             cudaMemcpyDeviceToHost, thandle));
#elif defined(USE_HIP)
  HIP_CHECK(hipMemcpyAsync(cinter_buf.data(), cinter_buf_dev, cinter_buf.size() * sizeof(T1),
                           hipMemcpyDeviceToHost, thandle));
#endif
}

template<typename T>
void allocate_device_buffers(ExecutionHW hw, T** dev_buf, size_t buf_size) {
  if(hw != ExecutionHW::GPU) return;
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
  auto& memPool = tamm::GPUPooledStorageManager::getInstance();
  *dev_buf      = static_cast<T*>(memPool.allocate(buf_size * sizeof(T)));
#endif
}

template<typename T>
void free_device_buffers(ExecutionHW hw, T* dev_buf, std::size_t buf_size) {
  if(hw != ExecutionHW::GPU) return;
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
  auto& memPool = tamm::GPUPooledStorageManager::getInstance();
  memPool.deallocate(static_cast<void*>(dev_buf), buf_size * sizeof(T));
#endif
}

template<typename T>
void assign_gpu(gpuStream_t& thandle, T* dst, const SizeVec& ddims, const IntLabelVec& dlabels,
                T scale, T* src, const SizeVec& sdims, const IntLabelVec& slabels, bool is_assign) {
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))

  const int ndim = sdims.size();

  const Size ssize = std::accumulate(sdims.begin(), sdims.end(), Size{1}, std::multiplies<Size>());
  if(ndim <= 1 || ssize.value() == 1) {
#if defined(USE_CUDA)
    CUDA_CHECK(
      cudaMemcpyAsync(dst, src, ssize.value() * sizeof(T), cudaMemcpyDeviceToDevice, thandle));
#elif defined(USE_HIP)
    HIP_CHECK(
      hipMemcpyAsync(dst, src, ssize.value() * sizeof(T), hipMemcpyDeviceToDevice, thandle));
#elif defined(USE_DPCPP)
    thandle.memcpy(dst, src, ssize.value() * sizeof(T));
#endif
    return;
  }

  std::vector<int> r_sdims;
  std::transform(std::begin(sdims), std::end(sdims), std::back_inserter(r_sdims),
                 [](tamm::Size i) -> int { return i.value(); });

  tamm::IntLabelVec r_dlabels = dlabels;
  tamm::IntLabelVec r_slabels = slabels;

  // if(is_assign)
  std::reverse(r_sdims.begin(), r_sdims.end());
  std::reverse(r_slabels.begin(), r_slabels.end());
  std::reverse(r_dlabels.begin(), r_dlabels.end());

  int perm[ndim];
  int size[ndim];
  // T beta         = is_assign ? 0 : 1;

  for(size_t i = 0; i < r_sdims.size(); i++) { size[i] = r_sdims[i]; }
  for(size_t i = 0; i < r_dlabels.size(); i++) {
    auto it = std::find(r_slabels.begin(), r_slabels.end(), r_dlabels[i]);
    EXPECTS(it != r_slabels.end());
    perm[i] = it - r_slabels.begin();
  }

  // create plan
  librettHandle plan;
#if defined(USE_DPCPP)
  librettPlan(&plan, ndim, size, perm, sizeof(T), &thandle);
#else
  librettPlan(&plan, ndim, size, perm, sizeof(T), thandle);
#endif
  librettExecute(plan, src, dst);
  librettDestroy(plan);

#endif
}

template<typename T2, typename T3>
bool transpose_inputs(bool& isgpuOp, gpuStream_t& thandle, std::vector<T2>& ainter_buf,
                      const SizeVec& ainter_dims, const IntLabelVec& ainter_labels, const T2* abuf,
                      size_t asize, const SizeVec& adims, const IntLabelVec& alabels,
                      std::vector<T3>& binter_buf, const SizeVec& binter_dims,
                      const IntLabelVec& binter_labels, const T3* bbuf, size_t bsize,
                      const SizeVec& bdims, const IntLabelVec& blabels, T2** ainter_buf_dev,
                      T3** binter_buf_dev) {
  bool gpu_trans = false;

#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
  if(isgpuOp) {
    gpu_trans = true;

    T2* ainter_buf_dev_in{nullptr};
    T3* binter_buf_dev_in{nullptr};

    auto& memPool = tamm::GPUPooledStorageManager::getInstance();

    ainter_buf_dev_in = static_cast<T2*>(memPool.allocate(asize * sizeof(T2)));
    binter_buf_dev_in = static_cast<T3*>(memPool.allocate(bsize * sizeof(T3)));

    copy_data_to_gpu_trans(isgpuOp, thandle, abuf, asize, &ainter_buf_dev_in, bbuf, bsize,
                           &binter_buf_dev_in);

    assign_gpu<T2>(thandle, *ainter_buf_dev, ainter_dims, ainter_labels, T2{1}, ainter_buf_dev_in,
                   adims, alabels, true);
    assign_gpu<T3>(thandle, *binter_buf_dev, binter_dims, binter_labels, T3{1}, binter_buf_dev_in,
                   bdims, blabels, true);

    memPool.deallocate(static_cast<void*>(ainter_buf_dev_in), asize * sizeof(T2));
    memPool.deallocate(static_cast<void*>(binter_buf_dev_in), bsize * sizeof(T3));

    return gpu_trans;
  }
#endif

  assign<T2>(ainter_buf.data(), ainter_dims, ainter_labels, T2{1}, abuf, adims, alabels, true);
  assign<T3>(binter_buf.data(), binter_dims, binter_labels, T3{1}, bbuf, bdims, blabels, true);
  return gpu_trans;
}

template<typename T1>
void transpose_output(bool& isgpuOp, gpuStream_t& thandle, bool gpu_trans,
                      std::vector<T1>& cinter_buf, const SizeVec& cinter_dims,
                      const IntLabelVec& cinter_labels, T1* cbuf, const SizeVec& cdims,
                      const IntLabelVec& clabels, T1** cinter_buf_dev, T1** cinter_tmp_buf_dev,
                      bool is_assign) {
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
  if(isgpuOp) {
    assign_gpu<T1>(thandle, *cinter_buf_dev, cdims, clabels, T1{1}, *cinter_tmp_buf_dev,
                   cinter_dims, cinter_labels, is_assign);
    return;
  }
#endif

  assign<T1>(cbuf, cdims, clabels, T1{1}, cinter_buf.data(), cinter_dims, cinter_labels, is_assign);
}

template<typename T, typename T1, typename T2, typename T3>
void block_multiply(bool& isgpuOp,
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
                    T2** th_a, T3** th_b,
#endif
                    gpuStream_t& thandle, T alpha, const T2* abuf, const SizeVec& adims,
                    const IntLabelVec& alabels, const T3* bbuf, const SizeVec& bdims,
                    const IntLabelVec& blabels, T beta, T1* cbuf, const SizeVec& cdims,
                    const IntLabelVec& clabels, ExecutionHW hw,
                    // bool has_gpu = false, bool is_assign = true
                    bool has_gpu, bool is_assign, T1** cinter_buf_dev, T1** cinter_tmp_buf_dev) {

  if(hw == ExecutionHW::GPU) isgpuOp = true;
  const Size asize = std::accumulate(adims.begin(), adims.end(), Size{1}, std::multiplies<Size>());
  const Size bsize = std::accumulate(bdims.begin(), bdims.end(), Size{1}, std::multiplies<Size>());
  const Size csize = std::accumulate(cdims.begin(), cdims.end(), Size{1}, std::multiplies<Size>());

  EXPECTS(abuf != nullptr && bbuf != nullptr && cbuf != nullptr);

  IntLabelVec asorted_labels{alabels}, bsorted_labels{blabels}, csorted_labels{clabels};
  std::sort(asorted_labels.begin(), asorted_labels.end());
  std::sort(bsorted_labels.begin(), bsorted_labels.end());
  std::sort(csorted_labels.begin(), csorted_labels.end());

  std::vector<IntLabel> inner_labels, aouter_labels, bouter_labels, batch_labels, areduce_labels,
    breduce_labels;
  std::vector<Size> inner_dims, aouter_dims, bouter_dims, batch_dims, areduce_dims, breduce_dims;

  int B = 1, M = 1, N = 1, K = 1, AR = 1, BR = 1;
  for(size_t i = 0; i < cdims.size(); i++) {
    const auto& lbl     = clabels[i];
    bool        is_in_a = std::binary_search(asorted_labels.begin(), asorted_labels.end(), lbl);
    bool        is_in_b = std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
    if(is_in_a && is_in_b) {
      batch_labels.push_back(lbl);
      batch_dims.push_back(cdims[i]);
      B *= static_cast<int>(cdims[i].value());
    }
    else if(is_in_a) {
      aouter_labels.push_back(lbl);
      aouter_dims.push_back(cdims[i]);
      M *= static_cast<int>(cdims[i].value());
    }
    else if(is_in_b) {
      bouter_labels.push_back(lbl);
      bouter_dims.push_back(cdims[i]);
      N *= static_cast<int>(cdims[i].value());
    }
    else {
      // UNREACHABLE();
    }
  }

  for(size_t i = 0; i < adims.size(); i++) {
    const auto& lbl     = alabels[i];
    bool        is_in_b = std::binary_search(bsorted_labels.begin(), bsorted_labels.end(), lbl);
    bool        is_in_c = std::binary_search(csorted_labels.begin(), csorted_labels.end(), lbl);
    if(is_in_b && is_in_c) {
      // no-op -- already added in batch_labels
    }
    else if(is_in_b) {
      inner_labels.push_back(lbl);
      inner_dims.push_back(adims[i]);
      K *= static_cast<int>(adims[i].value());
    }
    else if(is_in_c) {
      // no-op -- already added to aouter
    }
    else {
      AR *= adims[i].value();
      areduce_dims.push_back(adims[i]);
      areduce_labels.push_back(lbl);
    }
  }

  for(size_t i = 0; i < bdims.size(); i++) {
    const auto& lbl     = blabels[i];
    bool        is_in_a = std::binary_search(asorted_labels.begin(), asorted_labels.end(), lbl);
    bool        is_in_c = std::binary_search(csorted_labels.begin(), csorted_labels.end(), lbl);
    if(is_in_a && is_in_c) {
      // no-op -- already added in batch_labels
    }
    else if(is_in_a) {
      // no-op -- already in inner_labels
    }
    else if(is_in_c) {
      // no-op -- already added to bouter
    }
    else {
      BR *= bdims[i].value();
      breduce_dims.push_back(bdims[i]);
      breduce_labels.push_back(lbl);
    }
  }

  std::vector<IntLabel> ainter_labels{areduce_labels};
  ainter_labels.insert(ainter_labels.end(), batch_labels.begin(), batch_labels.end());
  ainter_labels.insert(ainter_labels.end(), aouter_labels.begin(), aouter_labels.end());
  ainter_labels.insert(ainter_labels.end(), inner_labels.begin(), inner_labels.end());

  std::vector<IntLabel> binter_labels{breduce_labels};
  binter_labels.insert(binter_labels.end(), batch_labels.begin(), batch_labels.end());
  binter_labels.insert(binter_labels.end(), inner_labels.begin(), inner_labels.end());
  binter_labels.insert(binter_labels.end(), bouter_labels.begin(), bouter_labels.end());

  std::vector<IntLabel> cinter_labels{batch_labels};
  cinter_labels.insert(cinter_labels.end(), aouter_labels.begin(), aouter_labels.end());
  cinter_labels.insert(cinter_labels.end(), bouter_labels.begin(), bouter_labels.end());

  SizeVec ainter_dims{areduce_dims};
  ainter_dims.insert(ainter_dims.end(), batch_dims.begin(), batch_dims.end());
  ainter_dims.insert(ainter_dims.end(), aouter_dims.begin(), aouter_dims.end());
  ainter_dims.insert(ainter_dims.end(), inner_dims.begin(), inner_dims.end());

  SizeVec binter_dims{breduce_dims};
  binter_dims.insert(binter_dims.end(), batch_dims.begin(), batch_dims.end());
  binter_dims.insert(binter_dims.end(), inner_dims.begin(), inner_dims.end());
  binter_dims.insert(binter_dims.end(), bouter_dims.begin(), bouter_dims.end());

  SizeVec cinter_dims{batch_dims};
  cinter_dims.insert(cinter_dims.end(), aouter_dims.begin(), aouter_dims.end());
  cinter_dims.insert(cinter_dims.end(), bouter_dims.begin(), bouter_dims.end());

  int ainter_ld  = K;
  int binter_ld  = N;
  int cinter_ld  = N;
  int cbatch_ld  = M * N;
  int abatch_ld  = M * K;
  int bbatch_ld  = K * N;
  int areduce_ld = B * abatch_ld;
  int breduce_ld = B * bbatch_ld;

  // optimization to run on CPU instead of GPU
  #if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
  if(isgpuOp && M < 1000 && N < 1000 && K < 1000) {
    isgpuOp = false;
    hw      = ExecutionHW::CPU;
  }
  #endif

  auto bmult_lambda = [&]() {
    bool            gpu_trans = false;
    std::vector<T1> cinter_buf;
    if(hw != ExecutionHW::GPU || !isgpuOp) cinter_buf.resize(static_cast<size_t>(csize.value()));

    T2* ainter_buf_dev{nullptr};
    T3* binter_buf_dev{nullptr};
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
    ainter_buf_dev = *th_a;
    binter_buf_dev = *th_b;
#endif

    // dgemm
    if constexpr(std::is_same_v<T1, T2> && std::is_same_v<T1, T3>) {
      std::vector<T2> ainter_buf;
      std::vector<T3> binter_buf;
      if(hw != ExecutionHW::GPU || !isgpuOp) {
        ainter_buf.resize(static_cast<size_t>(asize.value()));
        binter_buf.resize(static_cast<size_t>(bsize.value()));
      }

      gpu_trans = transpose_inputs(isgpuOp, thandle, ainter_buf, ainter_dims, ainter_labels, abuf,
                                   asize.value(), adims, alabels, binter_buf, binter_dims,
                                   binter_labels, bbuf, bsize.value(), bdims, blabels,
                                   &ainter_buf_dev, &binter_buf_dev);

      if(!gpu_trans)
        copy_data_to_gpu_trans(isgpuOp, thandle, ainter_buf.data(), ainter_buf.size(),
                               &ainter_buf_dev, binter_buf.data(), binter_buf.size(),
                               &binter_buf_dev);

      gemm_wrapper(isgpuOp, thandle, AR, BR, B, M, N, K, alpha, beta, ainter_buf, ainter_buf_dev,
                   binter_buf, binter_buf_dev, cinter_buf, *cinter_tmp_buf_dev);

      transpose_output(isgpuOp, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels, cbuf,
                       cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);
    }
    else {
      T2* abufp = const_cast<T2*>(abuf);
      T3* bbufp = const_cast<T3*>(bbuf);
      // TODO: actually check if one of T2, T3 is real, T1 is complex
      if constexpr(std::is_same_v<T1, T2>) {
        std::vector<T2> ainter_buf;
        std::vector<T1> binter_buf;
        if(hw != ExecutionHW::GPU || !isgpuOp) {
          ainter_buf.resize(static_cast<size_t>(asize.value()));
          binter_buf.resize(static_cast<size_t>(bsize.value()));
        }

        // T2 (matrix A) is complex, T3 (B) is real
        if constexpr(internal::is_complex_v<T1>) {
          // copy B to complex buffer
          std::vector<T1> bbuf_complex(bsize.value());
          T3*             bbuf_comp_ptr = reinterpret_cast<T3*>(bbuf_complex.data());
          blas::copy(bsize.value(), bbufp, 1, bbuf_comp_ptr, 2);

          T1* bbuf_complex_dev{nullptr};
          allocate_device_buffers(hw, &bbuf_complex_dev, bbuf_complex.size());

          gpu_trans = transpose_inputs(isgpuOp, thandle, ainter_buf, ainter_dims, ainter_labels,
                                       abuf, asize.value(), adims, alabels, binter_buf, binter_dims,
                                       binter_labels, bbuf_complex.data(), bsize.value(), bdims,
                                       blabels, &ainter_buf_dev, &bbuf_complex_dev);

          if(!gpu_trans) {
            bbuf_complex = binter_buf;
            copy_data_to_gpu(isgpuOp, thandle, ainter_buf, &ainter_buf_dev, bbuf_complex,
                             &bbuf_complex_dev);
          }

          gemm_wrapper(isgpuOp, thandle, AR, BR, B, M, N, K, alpha, beta, ainter_buf,
                       ainter_buf_dev, bbuf_complex, bbuf_complex_dev, cinter_buf,
                       *cinter_tmp_buf_dev);
          transpose_output(isgpuOp, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels,
                           cbuf, cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

          free_device_buffers(hw, bbuf_complex_dev, bbuf_complex.size());

        } // is_complex<T1>
        else {
          // T1,T2 (C,A) are real, T3 (B) is complex
          std::vector<T1> bbuf_real(bsize.value());
          T1*             bbuf_comp_ptr = reinterpret_cast<T1*>(bbufp);
          blas::copy(bsize.value(), bbuf_comp_ptr, 2, bbuf_real.data(), 1);

          T1* bbuf_real_dev{nullptr};
          allocate_device_buffers(hw, &bbuf_real_dev, bbuf_real.size());

          gpu_trans = transpose_inputs(isgpuOp, thandle, ainter_buf, ainter_dims, ainter_labels,
                                       abuf, asize.value(), adims, alabels, binter_buf, binter_dims,
                                       binter_labels, bbuf_real.data(), bsize.value(), bdims,
                                       blabels, &ainter_buf_dev, &bbuf_real_dev);

          if(!gpu_trans) {
            bbuf_real = binter_buf;
            copy_data_to_gpu(isgpuOp, thandle, ainter_buf, &ainter_buf_dev, bbuf_real,
                             &bbuf_real_dev);
          }

          gemm_wrapper(isgpuOp, thandle, AR, BR, B, M, N, K, alpha, beta, ainter_buf,
                       ainter_buf_dev, bbuf_real, bbuf_real_dev, cinter_buf, *cinter_tmp_buf_dev);
          transpose_output(isgpuOp, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels,
                           cbuf, cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

          free_device_buffers(hw, bbuf_real_dev, bbuf_real.size());
        } // is_real<T1>

      } // is_same_v<T1,T2>
      else if constexpr(std::is_same_v<T1, T3>) {
        std::vector<T1> ainter_buf;
        std::vector<T3> binter_buf;
        if(hw != ExecutionHW::GPU || !isgpuOp) {
          ainter_buf.resize(static_cast<size_t>(asize.value()));
          binter_buf.resize(static_cast<size_t>(bsize.value()));
        }

        // T3 (matrix B) is complex, T2 (A) is real
        if constexpr(internal::is_complex_v<T1>) {
          std::vector<T1> abuf_complex(asize.value());
          T2*             abuf_comp_ptr = reinterpret_cast<T2*>(abuf_complex.data());
          blas::copy(asize.value(), abufp, 1, abuf_comp_ptr, 2);

          T1* abuf_complex_dev{nullptr};
          allocate_device_buffers(hw, &abuf_complex_dev, abuf_complex.size());

          gpu_trans = transpose_inputs(isgpuOp, thandle, ainter_buf, ainter_dims, ainter_labels,
                                       abuf_complex.data(), asize.value(), adims, alabels,
                                       binter_buf, binter_dims, binter_labels, bbuf, bsize.value(),
                                       bdims, blabels, &abuf_complex_dev, &binter_buf_dev);

          if(!gpu_trans) {
            abuf_complex = ainter_buf;
            copy_data_to_gpu(isgpuOp, thandle, abuf_complex, &abuf_complex_dev, binter_buf,
                             &binter_buf_dev);
          }

          gemm_wrapper(isgpuOp, thandle, AR, BR, B, M, N, K, alpha, beta, abuf_complex,
                       abuf_complex_dev, binter_buf, binter_buf_dev, cinter_buf,
                       *cinter_tmp_buf_dev);
          transpose_output(isgpuOp, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels,
                           cbuf, cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

          free_device_buffers(hw, abuf_complex_dev, abuf_complex.size());
        }
        else {
          // T1,T3 (C,B) are real, T2 (A) is complex
          std::vector<T1> abuf_real(asize.value());

          T1* abuf_comp_ptr = reinterpret_cast<T1*>(abufp);
          blas::copy(asize.value(), abuf_comp_ptr, 2, abuf_real.data(), 1);

          T1* abuf_real_dev{nullptr};
          allocate_device_buffers(hw, &abuf_real_dev, abuf_real.size());

          gpu_trans = transpose_inputs(isgpuOp, thandle, ainter_buf, ainter_dims, ainter_labels,
                                       abuf_real.data(), asize.value(), adims, alabels, binter_buf,
                                       binter_dims, binter_labels, bbuf, bsize.value(), bdims,
                                       blabels, &abuf_real_dev, &binter_buf_dev);

          if(!gpu_trans) {
            abuf_real = ainter_buf;
            copy_data_to_gpu(isgpuOp, thandle, abuf_real, &abuf_real_dev, binter_buf,
                             &binter_buf_dev);
          }

          gemm_wrapper(isgpuOp, thandle, AR, BR, B, M, N, K, alpha, beta, abuf_real, abuf_real_dev,
                       binter_buf, binter_buf_dev, cinter_buf, *cinter_tmp_buf_dev);
          transpose_output(isgpuOp, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels,
                           cbuf, cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

          free_device_buffers(hw, abuf_real_dev, abuf_real.size());
        }

      } // is_same_v<T1,T3>

      else if constexpr(internal::is_complex_v<T1> && std::is_same_v<T2, T3>) {
        std::vector<T1> ainter_buf;
        std::vector<T1> binter_buf;
        if(hw != ExecutionHW::GPU || !isgpuOp) {
          ainter_buf.resize(static_cast<size_t>(asize.value()));
          binter_buf.resize(static_cast<size_t>(bsize.value()));
        }

        std::vector<T1> abuf_complex(asize.value());
        std::vector<T1> bbuf_complex(bsize.value());

        T2* abuf_comp_ptr = reinterpret_cast<T2*>(abuf_complex.data());
        T2* bbuf_comp_ptr = reinterpret_cast<T2*>(bbuf_complex.data());

        blas::copy(asize.value(), abufp, 1, abuf_comp_ptr, 2);
        blas::copy(bsize.value(), bbufp, 1, bbuf_comp_ptr, 2);

        T1* abuf_complex_dev{nullptr};
        T1* bbuf_complex_dev{nullptr};
        allocate_device_buffers(hw, &abuf_complex_dev, abuf_complex.size());
        allocate_device_buffers(hw, &bbuf_complex_dev, bbuf_complex.size());

        gpu_trans = transpose_inputs(isgpuOp, thandle, ainter_buf, ainter_dims, ainter_labels,
                                     abuf_complex.data(), asize.value(), adims, alabels, binter_buf,
                                     binter_dims, binter_labels, bbuf_complex.data(), bsize.value(),
                                     bdims, blabels, &abuf_complex_dev, &bbuf_complex_dev);

        if(!gpu_trans) {
          abuf_complex = ainter_buf;
          bbuf_complex = binter_buf;
          copy_data_to_gpu(isgpuOp, thandle, abuf_complex, &abuf_complex_dev, bbuf_complex,
                           &bbuf_complex_dev);
        }

        gemm_wrapper(isgpuOp, thandle, AR, BR, B, M, N, K, alpha, beta, abuf_complex,
                     abuf_complex_dev, bbuf_complex, bbuf_complex_dev, cinter_buf, *cinter_buf_dev);
        transpose_output(isgpuOp, thandle, gpu_trans, cinter_buf, cinter_dims, cinter_labels, cbuf,
                         cdims, clabels, cinter_buf_dev, cinter_tmp_buf_dev, is_assign);

        free_device_buffers(hw, abuf_complex_dev, abuf_complex.size());
        free_device_buffers(hw, bbuf_complex_dev, bbuf_complex.size());
      }

      else
        NOT_IMPLEMENTED();
    }

#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
    *th_a = ainter_buf_dev;
    *th_b = binter_buf_dev;
#endif

    if(is_assign && hw != ExecutionHW::GPU) // not using bufacc code path
      assign<T1>(cbuf, cdims, clabels, T{1}, cinter_buf.data(), cinter_dims, cinter_labels,
                 is_assign);
  };

  bmult_lambda();

} // block_multiply()

} // namespace kernels

} // namespace tamm
