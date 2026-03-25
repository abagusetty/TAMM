#pragma once

#include "device_memory_resource.hpp"

#include <cassert>
#include <cstddef>

namespace tamm::rmm::mr {

/**
 * @brief device_memory_resource that uses gpuMalloc/gpuFree.
 *
 * Changes vs. prior implementation:
 * - cudaFree / hipFree return codes are now asserted in debug builds.
 *   Silent failures masked double-frees and wrong-device errors that occur
 *   during reset_rmm_pool().
 * - do_deallocate is marked noexcept; GPU free functions must not throw.
 */
class gpu_memory_resource final : public device_memory_resource {
public:
  gpu_memory_resource()                                      = default;
  ~gpu_memory_resource() override                            = default;
  gpu_memory_resource(gpu_memory_resource const&)            = default;
  gpu_memory_resource(gpu_memory_resource&&)                 = default;
  gpu_memory_resource& operator=(gpu_memory_resource const&) = default;
  gpu_memory_resource& operator=(gpu_memory_resource&&)      = default;

private:
  void* do_allocate(std::size_t bytes) override {
    void* ptr{nullptr};
#if defined(USE_CUDA)
    auto status = cudaMalloc(&ptr, bytes);
    if(cudaSuccess != status) throw std::bad_alloc{};
#elif defined(USE_HIP)
    auto status = hipMalloc(&ptr, bytes);
    if(hipSuccess != status) throw std::bad_alloc{};
#elif defined(USE_DPCPP)
    ptr = sycl::malloc_device(bytes, GPUStreamPool::getInstance().getStream().first);
    if(ptr == nullptr) throw std::bad_alloc{};
#endif
    return ptr;
  }

  /**
   * @note bytes is unused by the GPU free functions but kept for interface
   *       consistency. [[maybe_unused]] suppresses the compiler warning.
   */
  void do_deallocate(void* ptr, [[maybe_unused]] std::size_t bytes) noexcept override {
#if defined(USE_CUDA)
    [[maybe_unused]] auto status = cudaFree(ptr);
    assert(cudaSuccess == status &&
           "cudaFree failed — possible double-free or wrong CUDA device context");
#elif defined(USE_HIP)
    [[maybe_unused]] auto status = hipFree(ptr);
    assert(hipSuccess == status &&
           "hipFree failed — possible double-free or wrong HIP device context");
#elif defined(USE_DPCPP)
    sycl::free(ptr, GPUStreamPool::getInstance().getStream().first);
#endif
  }

  [[nodiscard]] bool do_is_equal(
      device_memory_resource const& other) const noexcept override {
    return dynamic_cast<gpu_memory_resource const*>(&other) != nullptr;
  }
};

} // namespace tamm::rmm::mr
