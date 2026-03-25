#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace tamm::rmm::detail {

/**
 * @brief Default alignment used for CPU/GPU memory allocation.
 */
#if defined(USE_DPCPP)
static constexpr std::size_t RMM_ALLOCATION_ALIGNMENT{4};
#elif defined(USE_HIP)
static constexpr std::size_t RMM_ALLOCATION_ALIGNMENT{128};
#elif defined(USE_CUDA)
static constexpr std::size_t RMM_ALLOCATION_ALIGNMENT{256};
#else
static constexpr std::size_t RMM_ALLOCATION_ALIGNMENT{alignof(std::max_align_t)};
#endif

[[nodiscard]] constexpr bool is_pow2(std::size_t value) noexcept {
  return value != 0 && (0 == (value & (value - 1)));
}

[[nodiscard]] constexpr bool is_supported_alignment(std::size_t alignment) noexcept {
  return is_pow2(alignment);
}

/**
 * @brief Align up to nearest multiple of specified power of 2.
 *
 * Guards against std::size_t wraparound when value is near SIZE_MAX.
 */
[[nodiscard]] constexpr std::size_t align_up(std::size_t value,
                                              std::size_t alignment) noexcept {
  assert(is_supported_alignment(alignment));
  // Guard: if adding (alignment-1) would overflow size_t, saturate to max
  constexpr std::size_t max_v = std::numeric_limits<std::size_t>::max();
  if(value > max_v - (alignment - 1)) return max_v;
  return (value + (alignment - 1)) & ~(alignment - 1);
}

[[nodiscard]] constexpr std::size_t align_down(std::size_t value,
                                                std::size_t alignment) noexcept {
  assert(is_supported_alignment(alignment));
  return value & ~(alignment - 1);
}

[[nodiscard]] constexpr bool is_aligned(std::size_t value, std::size_t alignment) noexcept {
  assert(is_supported_alignment(alignment));
  return value == align_down(value, alignment);
}

[[nodiscard]] inline bool is_pointer_aligned(
    void* ptr, std::size_t alignment = RMM_ALLOCATION_ALIGNMENT) noexcept {
  return rmm::detail::is_aligned(
      reinterpret_cast<std::uintptr_t>(ptr), alignment); // use uintptr_t, not ptrdiff_t
}

/**
 * @brief Allocates sufficient memory to satisfy the requested size `bytes` with
 * alignment `alignment` using the unary callable `alloc` to allocate memory.
 *
 * The padded allocation size is: bytes + alignment + sizeof(std::ptrdiff_t).
 * The offset between the original and aligned pointer is stored at aligned-1.
 *
 * Allocations returned from `aligned_allocate` MUST be freed by calling
 * `aligned_deallocate` with the same `bytes` and `alignment` arguments.
 */
template<typename Alloc>
void* aligned_allocate(std::size_t bytes, std::size_t alignment, Alloc alloc) {
  assert(is_pow2(alignment));

  std::size_t const padded_allocation_size{bytes + alignment + sizeof(std::ptrdiff_t)};
  char* const original = static_cast<char*>(alloc(padded_allocation_size));

  void* aligned{original + sizeof(std::ptrdiff_t)};
  std::size_t space{padded_allocation_size - sizeof(std::ptrdiff_t)};
  std::align(alignment, bytes, aligned, space);

  // Store the offset between original and aligned pointer just before aligned
  std::ptrdiff_t const offset = static_cast<char*>(aligned) - original;
  *(static_cast<std::ptrdiff_t*>(aligned) - 1) = offset;

  return aligned;
}

/**
 * @brief Frees an allocation returned from `aligned_allocate`.
 *
 * The dealloc callable now receives BOTH the original pointer AND the
 * padded allocation size (bytes + alignment + sizeof(ptrdiff_t)), so that
 * size-aware allocators such as numa_free receive the correct byte count.
 *
 * @param ptr    The aligned pointer to deallocate.
 * @param bytes  The original (unpadded) size passed to aligned_allocate.
 * @param alignment  The alignment passed to aligned_allocate.
 * @param dealloc  Binary callable: dealloc(void* original, std::size_t padded).
 */
template<typename Dealloc>
void aligned_deallocate(void* ptr, std::size_t bytes, std::size_t alignment,
                        Dealloc dealloc) noexcept {
  // Recover the original pointer via stored offset
  std::ptrdiff_t const offset = *(reinterpret_cast<std::ptrdiff_t*>(ptr) - 1);
  void* const original        = static_cast<char*>(ptr) - offset;

  // Reconstruct the exact padded size used in aligned_allocate
  std::size_t const padded = bytes + alignment + sizeof(std::ptrdiff_t);

  dealloc(original, padded);
}

} // namespace tamm::rmm::detail
