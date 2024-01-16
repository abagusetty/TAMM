#include "tamm_blas.hpp"
#include <tamm/gpu_streams.hpp>

namespace tamm::gpu::kernel {

#ifdef USE_DPCPP
#define __global__ __attribute__((always_inline))
#endif

template<typename U, typename V>
constexpr auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

constexpr int TILE_DIM  = 32;
constexpr int THREADS_X = TILE_DIM;
constexpr int THREADS_Y = 256 / TILE_DIM;

template<typename T>
__global__ void reorderKernel(T* out, const T* in, const int* outDims, const int* inStrides,
                              const int* outStrides, const int d0, const int d1, const int d2,
                              const int d3, const int blocksPerMatX, const int blocksPerMatY
#ifdef USE_DPCPP
                              ,
                              sycl::nd_item<3> item
#endif
) {
#ifndef USE_DPCPP
  int threadIdx_x = threadIdx.x;
  int threadIdx_y = threadIdx.y;

  int blockDim_x = blockDim.x;
  int blockDim_y = blockDim.y;

  int blockIdx_x = blockIdx.x;
  int blockIdx_y = blockIdx.y;
  int blockIdx_z = blockIdx.z;

  int gridDim_y = gridDim.y;
#else
  int threadIdx_x = item.get_local_id(2);
  int threadIdx_y = item.get_local_id(1);

  int blockDim_x = item.get_local_range(2);
  int blockDim_y = item.get_local_range(1);

  int blockIdx_x = item.get_group(2);
  int blockIdx_y = item.get_group(1);
  int blockIdx_z = item.get_group(0);

  int gridDim_y = item.get_group_range(1);
#endif

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

// template<typename T, bool conjugate>
// __device__ T doOp(T in) {
//   if (conjugate)
//     return conj(in);
//   else
//     return in;
// }

//   template<typename T>
//   __global__ void transposeInPlaceKernel(T* in_ptr,
//                                          int* in_dims,
//                                          int* in_strides,
//                                          const int blocksPerMatX,
//                                          const int blocksPerMatY,
//                                          const bool conjugate,
//                                          const bool is32Multiple
// #ifdef USE_DPCPP
//                                          , sycl::nd_item<3> it
// #endif
//                                          ) {
//     #ifdef USE_DPCPP
//     sycl::group g = it.get_group();
//     using tile_t = T[TILE_DIM][TILE_DIM + 1];
//     tile_t& shrdMem_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(g);
//     tile_t& shrdMem_d = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(g);

//     const int lx = it.get_local_id(2);
//     const int ly = it.get_local_id(1);

//     // batch based block Id
//     const int batchId_x  = g.get_group_id(2) / blocksPerMatX;
//     const int blockIdx_x = (g.get_group_id(2) - batchId_x * blocksPerMatX);

//     const int batchId_y  = g.get_group_id(1) / blocksPerMatY;
//     const int blockIdx_y = (g.get_group_id(1) - batchId_y * blocksPerMatY);
//     #else
//     __shared__ T shrdMem_s[TILE_DIM][TILE_DIM + 1];
//     __shared__ T shrdMem_d[TILE_DIM][TILE_DIM + 1];

//     const int lx = threadIdx.x;
//     const int ly = threadIdx.y;

//     // batch based block Id
//     const int batchId_x  = blockIdx.x / blocksPerMatX;
//     const int blockIdx_x = (blockIdx.x - batchId_x * blocksPerMatX);

//     const int batchId_y  = blockIdx.y / blocksPerMatY;
//     const int blockIdx_y = (blockIdx.y - batchId_y * blocksPerMatY);
//     #endif

//     // create variables to hold output dimensions
//     const int iDim0 = in_dims[0];
//     const int iDim1 = in_dims[1];

//     // calculate strides
//     const int iStride1 = in_strides[1];

//     const int x0 = TILE_DIM * blockIdx_x;
//     const int y0 = TILE_DIM * blockIdx_y;

//     // offset in and out based on batch id
//     T *iptr = in_ptr + batchId_x * in_strides[2] + batchId_y * in_strides[3];

//     if (blockIdx_y > blockIdx_x) { // Off diagonal blocks
//       // calculate global indices
//       int gx = lx + x0;
//       int gy = ly + y0;
//       int dx = lx + y0;
//       int dy = ly + x0;

//       // Copy to shared memory
//       #pragma unroll
//       for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
//         int gy_ = gy + repeat;
//         if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
//           shrdMem_s[ly + repeat][lx] = iptr[gy_ * iStride1 + gx];

//         int dy_ = dy + repeat;
//         if (is32Multiple || (dx < iDim0 && dy_ < iDim1))
//           shrdMem_d[ly + repeat][lx] = iptr[dy_ * iStride1 + dx];
//       }

// #ifdef USE_DPCPP
//       sycl::group_barrier(g);
// #else
//       __syncthreads();
// #endif

//       // Copy from shared memory to global memory
//       #pragma unroll
//       for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
//         int dy_ = dy + repeat;
//         if (is32Multiple || (dx < iDim0 && dy_ < iDim1))
//           iptr[dy_ * iStride1 + dx] = doOp<T, conjugate>(shrdMem_s[lx][ly + repeat]);

//         int gy_ = gy + repeat;
//         if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
//           iptr[gy_ * iStride1 + gx] = doOp<T, conjugate>(shrdMem_d[lx][ly + repeat]);
//       }

//     } else if (blockIdx_y == blockIdx_x) { // Diagonal blocks
//       // calculate global indices
//       int gx = lx + x0;
//       int gy = ly + y0;

//       // offset in and out based on batch id
//       iptr = in_ptr + batchId_x * in_strides[2] + batchId_y * in_strides[3];

//       // Copy to shared memory
//       #pragma unroll
//       for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
//         int gy_ = gy + repeat;
//         if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
//           shrdMem_s[ly + repeat][lx] = iptr[gy_ * iStride1 + gx];
//       }

// #ifdef USE_DPCPP
//       sycl::group_barrier(g);
// #else
//       __syncthreads();
// #endif

//       // Copy from shared memory to global memory
//       for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
//         int gy_ = gy + repeat;
//         if (is32Multiple || (gx < iDim0 && gy_ < iDim1))
//           iptr[gy_ * iStride1 + gx] = doOp<T, conjugate>(shrdMem_s[lx][ly + repeat]);
//       }
//     }
//   }

template<typename T>
void transpose_inplace(T* out, const T* in, int* outDims, int* rdims, gpuStream_t& thandle) {
  // Step 1: Launch reorder
  constexpr unsigned TX    = 32;
  constexpr unsigned TY    = 8;
  constexpr unsigned TILEX = 512;
  constexpr unsigned TILEY = 32;

  int blocksPerMatX = divUp(outDims[0], TILEX);
  int blocksPerMatY = divUp(outDims[1], TILEY);

#ifndef USE_DPCPP

  dim3 threads(TX, TY, 1);
  dim3 blocks(blocksPerMatX * outDims[2], blocksPerMatY * outDims[3], 1);

  const int maxBlocksY = getDeviceProp(getActiveDeviceId()).maxGridSize[1];
  blocks.z             = divUp(blocks.y, maxBlocksY);
  blocks.y             = divUp(blocks.y, blocks.z);

  reorderKernel<<<blocks, threads, 0, thandle.first>>>(
    out, in, outDims, rdims[0], rdims[1], rdims[2], rdims[3], blocksPerMatX, blocksPerMatY);

#else

  auto local = sycl::range{1, TY, TX};

  sycl::id<3> max_num_wrk_groups =
    thandle.first.get_device()
      .get_info<sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>();
  auto      blocks     = sycl::range(1, blocksPerMatY * outDims[3], blocksPerMatX * outDims[2]);
  const int maxBlocksY = max_num_wrk_groups[1];
  blocks[0]            = divUp(blocks[1], maxBlocksY);
  blocks[1]            = divUp(blocks[1], blocks[0]);

  auto global = blocks * local;

  int stride[4] = {1, 1, 1, 1};
  thandle.first.parallel_for(sycl::nd_range<3>(global, local), [=](auto item) {
    reorderKernel<T>(out, in, outDims, stride, stride, rdims[0], rdims[1], rdims[2], rdims[3],
                     blocksPerMatX, blocksPerMatY, item);
  });

#endif

  //     // Step 2: Launch transpose
  //     const bool is32Multiple = inDims[0] % TILE_DIM == 0 && inDims[1] % TILE_DIM == 0;

  //     int blk_x = divUp(inDims[0], TILE_DIM);
  //     int blk_y = divUp(inDims[1], TILE_DIM);

  // #ifndef USE_DPCPP
  //     threads(THREADS_X, THREADS_Y);
  //     blocks(blk_x * inDims[2], blk_y * inDims[3]);

  //     transposeInPlaceKernel<T><<<blocks, threads, 0, thandle.first>>>(out, outDims, in.info,
  //     blk_x, blk_y, conjugate, is32Multiple);
  // #else
  //     local = sycl::range{1, THREADS_Y, THREADS_X};
  //     global = sycl::range(1, blk_y * THREADS_Y * inDims[3], blk_x * THREADS_X * inDims[2]);

  //     int outStride[4] = {1,1,1,1};
  //     thandle.first.parallel_for(sycl::nd_range<3>(global, local), [=](auto item) {
  //       transposeInPlaceKernel<T>(out, outDims, outStride, blk_x, blk_y, conjugate, is32Multiple,
  //       item);
  //     });
  // #endif
}

template void transpose_inplace(double* out, const double* in, int* outDims, int* inDims,
                                int* rdims, const bool conjugate, gpuStream_t& thandle);
// template void transpose_inplace(std::complex<double>* out, const std::complex<double>* in, int*
// outDims, int* inDims, int* rdims, const bool conjugate, gpuStream_t& thandle);

} // namespace tamm::gpu::kernel
