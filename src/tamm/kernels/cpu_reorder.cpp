#include "tamm_blas.hpp"

template<typename T>
void tamm::kernels::cpu::transpose_reorder(T*& outPtr, const T* inPtr, const int* outDims,
                                           const int* rdims) {
  int ids[4] = {0};
  int ost[4] = {1, 1, 1, 1};
  int ist[4] = {1, 1, 1, 1};

  for (int ow = 0; ow < outDims[3]; ow++) {
    const int oW = ow * ost[3];
    ids[rdims[3]]  = ow;
    for (int oz = 0; oz < outDims[2]; oz++) {
      const int oZW = oW + oz * ost[2];
      ids[rdims[2]]   = oz;
      for (int oy = 0; oy < outDims[1]; oy++) {
        const int oYZW = oZW + oy * ost[1];
        ids[rdims[1]]    = oy;
        for (int ox = 0; ox < outDims[0]; ox++) {
          const int oIdx = oYZW + ox;

          ids[rdims[0]]    = ox;
          const int iIdx = ids[3] * ist[3] + ids[2] * ist[2] +
            ids[1] * ist[1] + ids[0];

          outPtr[oIdx] = inPtr[iIdx];
        }
      }
    }
  }
}

// Explicit template instantiations
template void tamm::kernels::cpu::transpose_reorder(double*& out, const double* in, const int* outDims, const int* rdims);
template void tamm::kernels::cpu::transpose_reorder(std::complex<double>*& out, const std::complex<double>* in, const int* outDims, const int* rdims);
