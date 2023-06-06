#include "tamm_blas.hpp"

#include "ga/ga_linalg.h"

namespace tamm::kernels {
namespace cpu {

template<typename T, typename T1, typename T2, typename T3>
void blas(int m, int n, int k, const T alpha, const T2* A, int lda, const T3* B, int ldb,
          const T beta, T1* C, int ldc) {
  blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k, alpha, A, lda,
             B, ldb, beta, C, ldc);
}

} // namespace cpu
} // namespace tamm::kernels
