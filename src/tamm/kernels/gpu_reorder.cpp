#include "tamm_blas.hpp"
#include <tamm/gpu_streams.hpp>

namespace tamm {

template<typename U, typename V>
constexpr auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

constexpr int TILE_DIM  = 32;
constexpr int THREADS_X = TILE_DIM;
constexpr int THREADS_Y = 256 / TILE_DIM;

#ifdef USE_DPCPP
template<typename T>
__attribute__((always_inline)) void reorder(T* out, const T* in, const int outDims0, const int outDims1,
                                            const int outDims2, const int outDims3,
                                            const int d0, const int d1, const int d2, const int d3,
                                            const int blocksPerMatX, const int blocksPerMatY,
                                            sycl::nd_item<2> item
                                            ) {
  const int inStrides[4] = {1, 1, 1, 1};
  const int outStrides[4] = {1, 1, 1, 1};

  sycl::group g = item.get_group();

  const int oz = g.get_group_id(0) / blocksPerMatX;
  const int ow = g.get_group_id(1) / blocksPerMatY;

  const int blockIdx_x = g.get_group_id(0) - oz * blocksPerMatX;
  const int blockIdx_y = g.get_group_id(1) - ow * blocksPerMatY;

  const int xx = item.get_local_id(0) + blockIdx_x * g.get_local_range(0);
  const int yy = item.get_local_id(1) + blockIdx_y * g.get_local_range(1);

  bool valid = (xx < outDims0 && yy < outDims1 &&
                oz < outDims2 && ow < outDims3);

  const int incy = blocksPerMatY * g.get_local_range(1);
  const int incx = blocksPerMatX * g.get_local_range(0);

  const int o_off    = ow * outStrides[3] + oz * outStrides[2];
  const int rdims[4] = {d0, d1, d2, d3};
  int ids[4]         = {0};

  ids[rdims[3]] = ow;
  ids[rdims[2]] = oz;

  for (int oy = yy; oy < outDims1; oy += incy) {
    ids[rdims[1]] = oy;
    for (int ox = xx; ox < outDims0; ox += incx) {
      ids[rdims[0]] = ox;

      const int oIdx = o_off + oy * outStrides[1] + ox;

      const int iIdx = ids[3] * inStrides[3] +
        ids[2] * inStrides[2] +
        ids[1] * inStrides[1] + ids[0];

      if (valid) { out[oIdx] = in[iIdx]; }
    }
  }
}

#else

  template<typename T>
__global__ void reorder(T* out, const T* in, const int* outDims,
                        const int d0, const int d1, const int d2,
                        const int d3, const int blocksPerMatX, const int blocksPerMatY) {
  int threadIdx_x = threadIdx.x;
  int threadIdx_y = threadIdx.y;

  int blockDim_x = blockDim.x;
  int blockDim_y = blockDim.y;

  int blockIdx_x = blockIdx.x;
  int blockIdx_y = blockIdx.y;
  int blockIdx_z = blockIdx.z;

  int gridDim_y = gridDim.y;

  const int oz = blockIdx_x / blocksPerMatX;
  const int ow = (blockIdx_y + blockIdx_z * gridDim_y) / blocksPerMatY;

  const int blockIdx_xx = blockIdx_x - oz * blocksPerMatX;
  const int blockIdx_yy = (blockIdx_y + blockIdx_z * gridDim_y) - ow * blocksPerMatY;

  const int xx = threadIdx_x + blockIdx_xx * blockDim_x;
  const int yy = threadIdx_y + blockIdx_yy * blockDim_y;

  if(xx >= outDims[0] || yy >= outDims[1] || oz >= outDims[2] || ow >= outDims[3]) return;

  const int incy = blocksPerMatY * blockDim_y;
  const int incx = blocksPerMatX * blockDim_x;

  const int rdims[] = {d0, d1, d2, d3};
  const int o_off   = ow * outStrides[3] + oz * outStrides[2];
  int       ids[4]  = {0};
  ids[rdims[3]]     = ow;
  ids[rdims[2]]     = oz;

  for(int oy = yy; oy < outDims[1]; oy += incy) {
    ids[rdims[1]] = oy;
    for(int ox = xx; ox < outDims[0]; ox += incx) {
      ids[rdims[0]] = ox;

      const int oIdx = o_off + oy * outStrides[1] + ox;

      const int iIdx =
        ids[3] * inStrides[3] + ids[2] * inStrides[2] + ids[1] * inStrides[1] + ids[0];

      out[oIdx] = in[iIdx];
    }
  }
}
#endif

template<typename T>
void tamm::kernels::gpu::transpose_reorder(T*& out, const T* in, const int* outDims, const int* rdims, gpuStream_t& thandle) {
  constexpr unsigned TX    = 32;
  constexpr unsigned TY    = 8;
  constexpr unsigned TILEX = 512;
  constexpr unsigned TILEY = 32;

  // int blocksPerMatX = divUp(outDims[0], TILEX);
  // int blocksPerMatY = divUp(outDims[1], TILEY);
  const int maxBlocksY = tamm::getMaxGridSizeY();

#ifndef USE_DPCPP

  dim3 threads(TX, TY, 1);
  dim3 blocks(blocksPerMatX * outDims[2], blocksPerMatY * outDims[3], 1);

  blocks.z             = divUp(blocks.y, maxBlocksY);
  blocks.y             = divUp(blocks.y, blocks.z);

  reorder<<<blocks, threads, 0, thandle.first>>>(
    out, in, outDims, rdims[0], rdims[1], rdims[2], rdims[3], blocksPerMatX, blocksPerMatY);

#else

  // auto local = sycl::range{1, TY, TX};
  // auto blocks = sycl::range(1, blocksPerMatY * outDims[3], blocksPerMatX * outDims[2]);
  // blocks[0]            = divUp(blocks[1], maxBlocksY);
  // blocks[1]            = divUp(blocks[1], blocks[0]);

  auto local = sycl::range(TX, TY);
  int blocksPerMatX = divUp(outDims[0], TILEX);
  int blocksPerMatY = divUp(outDims[1], TILEY);
  auto global       = sycl::range(local[0] * blocksPerMatX * outDims[2],
                                  local[1] * blocksPerMatY * outDims[3]);
  
  // auto global = blocks * local;

  thandle.first.parallel_for(sycl::nd_range<2>(global, local), [=](auto item) {
    reorder<T>(out, in, outDims[0], outDims[1], outDims[2], outDims[3], rdims[0], rdims[1], rdims[2], rdims[3],
               blocksPerMatX, blocksPerMatY, item);
  });

#endif
}

template void tamm::kernels::gpu::transpose_reorder(double*& out, const double* in, const int* outDims, const int* rdims, gpuStream_t& thandle);
template void tamm::kernels::gpu::transpose_reorder(std::complex<double>*& out, const std::complex<double>* in, const int* outDims, const int* rdims, gpuStream_t& thandle);

} // namespace tamm
