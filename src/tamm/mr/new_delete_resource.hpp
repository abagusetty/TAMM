#pragma once

#include "aligned.hpp"
#include "host_memory_resource.hpp"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(TAMM_DISABLE_LIBNUMA)
#include <sys/sysinfo.h>
#else
#include <numa.h>
#endif

namespace tamm::rmm::mr {

/**
 * @brief host_memory_resource backed by operator new/delete (or numa_alloc).
 *
 * Bug fixed vs. prior implementation:
 *   numa_free(ptr, size) requires the PADDED allocation size
 *   (bytes + alignment + sizeof(ptrdiff_t)), not the original requested bytes.
 *   Previously the lambda captured `bytes` (pool-requested size), causing
 *   numa_free to under-report the freed region — a CPU memory accounting leak
 *   on all Linux nodes using libnuma.
 *
 *   The updated aligned_deallocate signature passes (void* original,
 *   std::size_t padded) to the dealloc callable, so the lambda now receives
 *   the correct padded size automatically.
 */
class new_delete_resource final : public host_memory_resource {
public:
  new_delete_resource()                                      = default;
  ~new_delete_resource() override                            = default;
  new_delete_resource(new_delete_resource const&)            = default;
  new_delete_resource(new_delete_resource&&)                 = default;
  new_delete_resource& operator=(new_delete_resource const&) = default;
  new_delete_resource& operator=(new_delete_resource&&)      = default;

private:
  void* do_allocate(std::size_t bytes,
                    std::size_t alignment = rmm::detail::RMM_ALLOCATION_ALIGNMENT) override {
    alignment = rmm::detail::is_supported_alignment(alignment)
                  ? alignment
                  : rmm::detail::RMM_ALLOCATION_ALIGNMENT;

#if defined(__APPLE__) || defined(TAMM_DISABLE_LIBNUMA)
    return rmm::detail::aligned_allocate(
      bytes, alignment,
      [](std::size_t size) { return ::operator new(size); });
#else
    return rmm::detail::aligned_allocate(
      bytes, alignment,
      [](std::size_t size) { return numa_alloc_onnode(size, numa_preferred()); });
#endif
  }

  void do_deallocate(void* ptr, std::size_t bytes,
                     std::size_t alignment = rmm::detail::RMM_ALLOCATION_ALIGNMENT) override {
#if defined(__APPLE__) || defined(TAMM_DISABLE_LIBNUMA)
    // operator delete does not need the size; padded is ignored
    rmm::detail::aligned_deallocate(
      ptr, bytes, alignment,
      [](void* p, std::size_t /*padded*/) { ::operator delete(p); });
#else
    // numa_free REQUIRES the exact size that was passed to numa_alloc_onnode.
    // aligned_deallocate now forwards padded = bytes+alignment+sizeof(ptrdiff_t)
    // so numa_free accounts for the full allocation correctly.
    rmm::detail::aligned_deallocate(
      ptr, bytes, alignment,
      [](void* p, std::size_t padded) { numa_free(p, padded); });
#endif
  }
};

} // namespace tamm::rmm::mr
