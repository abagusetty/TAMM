#include "tamm_blas.hpp"

#if defined(USE_CUDA)
#include <cublas_v2.h>
#elif defined(USE_HIP)
#include <rocblas.h>
#elif defined(USE_DPCPP)
#include <oneapi/mkl/blas.hpp>

#if defined(USE_SYCL_BLAS)
#include <sycl_blas.h>
#endif // USE_SYCL_BLAS

#endif // USE_DPCPP

#if defined(USE_HIP)
#define ROCBLAS_CHECK(FUNC)                                                                      \
  do {                                                                                           \
    rocblas_status err_ = (FUNC);                                                                \
    if(err_ != rocblas_status_success) {                                                         \
      std::ostringstream msg;                                                                    \
      msg << "ROCBLAS Error: " << rocblas_status_to_string(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                              \
      throw std::runtime_error(msg.str());                                                       \
    }                                                                                            \
  } while(0)
#endif // USE_HIP

#if defined(USE_CUDA)
#define CUBLAS_CHECK(FUNC)                                                                   \
  do {                                                                                       \
    cublasStatus_t err_ = (FUNC);                                                            \
    if(err_ != CUBLAS_STATUS_SUCCESS) {                                                      \
      std::ostringstream msg;                                                                \
      msg << "CUBLAS Error: " << cublasGetStatusString(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                          \
      throw std::runtime_error(msg.str());                                                   \
    }                                                                                        \
  } while(0)
#endif // USE_CUDA

#if defined(USE_DPCPP)
#define ONEMKLBLAS_CHECK(FUNC)                                                         \
  do {                                                                                 \
    try {                                                                              \
      (FUNC)                                                                           \
    } catch(oneapi::mkl::exception const& ex) {                                        \
      std::ostringstream msg;                                                          \
      msg << "oneMKL Error: " << ex.what() << ", at " << __FILE__ << " : " << __LINE__ \
          << std::endl;                                                                \
      throw std::runtime_error(msg.str());                                             \
    }                                                                                  \
  } while(0)
#endif // USE_DPCPP

namespace tamm::kernels {
namespace gpu {

template<typename T, typename T1, typename T2, typename T3>
void blas(int m, int n, int k, const T alpha, const T2* A, int lda, const T3* B, int ldb,
          const T beta, T1* C, int ldc, gpuStream_t& handle) {
#if defined(USE_DPCPP)
#ifdef USE_SYCL_BLAS
  auto dgemm = ::_gemm(handle.first, 'n', 'n', n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
  dgemm.wait();
#else
  try {
    auto dgemm = oneapi::mkl::blas::column_major::gemm(handle.first, oneapi::mkl::transpose::N,
                                                       oneapi::mkl::transpose::N, n, m, k, alpha, B,
                                                       ldb, A, lda, beta, C, ldc);
    dgemm.wait();
  } catch(oneapi::mkl::exception const& ex) {
    std::stringstream msg;
    msg << "oneMKL Exception at " << __FILE__ << " : " << __LINE__ << std::endl;
    throw(std::runtime_error(ex.what()));
  }
#endif // USE_SYCL_BLAS

#elif defined(USE_CUDA)
  if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
               internal::is_complex_v<T3>) {
    CUBLAS_CHECK(cublasZgemm(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             (cuDoubleComplex*) &alpha, (cuDoubleComplex*) B, ldb,
                             (cuDoubleComplex*) A, lda, (cuDoubleComplex*) &beta,
                             (cuDoubleComplex*) C, ldc));
  }
  else {
    CUBLAS_CHECK(cublasDgemm(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, ldb, A,
                             lda, &beta, C, ldc));
  }
#elif defined(USE_HIP)
  if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
               internal::is_complex_v<T3>) {
    ROCBLAS_CHECK(rocblas_zgemm(handle.second, rocblas_operation_none, rocblas_operation_none, n, m,
                                k, (rocblas_double_complex*) &alpha, (rocblas_double_complex*) B,
                                ldb, (rocblas_double_complex*) A, lda,
                                (rocblas_double_complex*) &beta, (rocblas_double_complex*) C, ldc));
  }
  else {
    ROCBLAS_CHECK(rocblas_dgemm(handle.second, rocblas_operation_none, rocblas_operation_none, n, m,
                                k, &alpha, B, ldb, A, lda, &beta, C, ldc));
  }
#endif
}

} // namespace gpu
} // namespace tamm::kernels
