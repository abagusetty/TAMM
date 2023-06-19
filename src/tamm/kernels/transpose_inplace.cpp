#include <tamm/gpu_streams.hpp>

namespace tamm {
namespace gpu {
namespace kernel {

constexpr dim_t TILE_DIM  = 32;
constexpr dim_t THREADS_X = TILE_DIM;
constexpr dim_t THREADS_Y = 256 / TILE_DIM;


  template<typename T>
  __global__ void transposeIP(T* in_ptr,
                              const int blocksPerMatX,
                              const int blocksPerMatY,
                              bool is32multiple_,
                              #ifdef USE_DPCPP
                              , sycl::nd_item<3> it
                              #endif
                              ) {
    #ifdef USE_DPCPP
    sycl::group g = it.get_group();
    using tile_t = T[TILE_DIM][TILE_DIM + 1];
    tile_t& shrdMem_s = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(g);
    tile_t& shrdMem_d = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(g);

    const int lx = it.get_local_id(2);
    const int ly = it.get_local_id(1);

    // batch based block Id
    const int batchId_x  = g.get_group_id(2) / blocksPerMatX;
    const int blockIdx_x = (g.get_group_id(2) - batchId_x * blocksPerMatX);

    const int batchId_y  = g.get_group_id(1) / blocksPerMatY;
    const int blockIdx_y = (g.get_group_id(1) - batchId_y * blocksPerMatY);
    #else
    __shared__ T shrdMem_s[TILE_DIM][TILE_DIM + 1];
    __shared__ T shrdMem_d[TILE_DIM][TILE_DIM + 1];

    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    // batch based block Id
    const int batchId_x  = blockIdx.x / blocksPerMatX;
    const int blockIdx_x = (blockIdx.x - batchId_x * blocksPerMatX);

    const int batchId_y  = blockIdx.y / blocksPerMatY;
    const int blockIdx_y = (blockIdx.y - batchId_y * blocksPerMatY);
    #endif

    // create variables to hold output dimensions
    const int iDim0 = in.dims[0];
    const int iDim1 = in.dims[1];

    // calculate strides
    const int iStride1 = in.strides[1];

    const int x0 = TILE_DIM * blockIdx_x;
    const int y0 = TILE_DIM * blockIdx_y;

    // offset in and out based on batch id
    T *iptr = in_ptr + batchId_x * in.strides[2] + batchId_y * in.strides[3];

    if (blockIdx_y > blockIdx_x) { // Off diagonal blocks
      // calculate global indices
      int gx = lx + x0;
      int gy = ly + y0;
      int dx = lx + y0;
      int dy = ly + x0;

      // Copy to shared memory
      #pragma unroll
      for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        int gy_ = gy + repeat;
        if (is32multiple_ || (gx < iDim0 && gy_ < iDim1))
          shrdMem_s[ly + repeat][lx] = iptr[gy_ * iStride1 + gx];

        int dy_ = dy + repeat;
        if (is32multiple_ || (dx < iDim0 && dy_ < iDim1))
          shrdMem_d[ly + repeat][lx] = iptr[dy_ * iStride1 + dx];
      }

#ifdef USE_DPCPP
      sycl::group_barrier(g);
#else
      __syncthreads();
#endif

      // Copy from shared memory to global memory
      #pragma unroll
      for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        int dy_ = dy + repeat;
        if (is32multiple_ || (dx < iDim0 && dy_ < iDim1))
          iptr[dy_ * iStride1 + dx] = shrdMem_s[lx][ly + repeat];

        int gy_ = gy + repeat;
        if (is32multiple_ || (gx < iDim0 && gy_ < iDim1))
          iptr[gy_ * iStride1 + gx] = shrdMem_d[lx][ly + repeat];
      }

    } else if (blockIdx_y == blockIdx_x) { // Diagonal blocks
      // calculate global indices
      int gx = lx + x0;
      int gy = ly + y0;

      // offset in and out based on batch id
      iptr = in_ptr + batchId_x * in.strides[2] + batchId_y * in.strides[3];

      // Copy to shared memory
      #pragma unroll
      for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        int gy_ = gy + repeat;
        if (is32multiple_ || (gx < iDim0 && gy_ < iDim1))
          shrdMem_s[ly + repeat][lx] = iptr[gy_ * iStride1 + gx];
      }

#ifdef USE_DPCPP
      sycl::group_barrier(g);      
#else
      __syncthreads();
#endif

      // Copy from shared memory to global memory
      for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
        int gy_ = gy + repeat;
        if (is32multiple_ || (gx < iDim0 && gy_ < iDim1))
          iptr[gy_ * iStride1 + gx] = shrdMem_s[lx][ly + repeat];
      }
    }
  }

template<typename T>
void transpose_inplace(T* in, int* inDims, const bool conjugate,
                       gpuStream_t& thandle) {
    const bool is32multiple = inDims[0] % TILE_DIM == 0 && inDims[1] % TILE_DIM == 0;
    
    int blk_x = divup(in.info.dims[0], TILE_DIM);
    int blk_y = divup(in.info.dims[1], TILE_DIM);
    
    #ifndef USE_DPCPP
    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks(blk_x * in.dims[2], blk_y * in.dims[3]);

    transposeInPlaceKernel<T><<<blocks, threads, 0, thandle>>>(r, in.info, blk_x, blk_y, conjugate,
                                    is32multiple);    
    #else
    auto local = sycl::range{THREADS_X, THREADS_Y};
    auto global = sycl::range{blk_x * local[0] * in.info.dims[2],
                              blk_y * local[1] * in.info.dims[3]};

    thandle.parallel_for(sycl::nd_range{global, local},
                            transposeInPlaceKernel<T>(r, in.info, blk_x, blk_y, conjugate,
                                                      is32multiple, item));
    #endif
}

}  // namespace kernel
}  // namespace gpu
}  // namespace tamm
