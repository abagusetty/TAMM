#pragma once

#include "aligned.hpp"
#include "device_memory_resource.hpp"

#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>

#ifndef TAMM_ENABLE_POOL_SANITY_CHECKS
// Enable expensive allocation tracking by default in debug builds. Define
// TAMM_ENABLE_POOL_SANITY_CHECKS to override at compile time.
#ifdef NDEBUG
#define TAMM_ENABLE_POOL_SANITY_CHECKS 0
#else
#define TAMM_ENABLE_POOL_SANITY_CHECKS 1
#endif
#endif

namespace tamm::rmm::mr::detail {

/**
 * @brief A CRTP helper function
 *
 * https://www.fluentcpp.com/2017/05/19/crtp-helper/
 *
 * Does two things:
 * 1. Makes "crtp" explicit in the inheritance structure of a CRTP base class.
 * 2. Avoids having to `static_cast` in a lot of places
 *
 * @tparam T The derived class in a CRTP hierarchy
 */
template<typename T>
struct crtp {
  [[nodiscard]] T&       underlying() { return static_cast<T&>(*this); }
  [[nodiscard]] T const& underlying() const { return static_cast<T const&>(*this); }
};

/**
 * @brief Base class for a stream-ordered memory resource
 *
 * This base class uses CRTP (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
 * to provide static polymorphism to enable defining suballocator resources that maintain separate
 * pools per stream. All of the stream-ordering logic is contained in this class, but the logic
 * to determine how memory pools are managed and the type of allocation is implented in a derived
 * class and in a free list class.
 *
 * For example, a coalescing pool memory resource uses a coalescing_free_list and maintains data
 * structures for allocated blocks and has functions to allocate and free blocks and to expand the
 * pool.
 *
 * Classes derived from stream_ordered_memory_resource must implement the following four methods,
 * documented separately:
 *
 * 1. `std::size_t get_maximum_allocation_size() const`
 * 2. `block_type expand_pool(std::size_t size, free_list& blocks, gpu_stream_view stream)`
 * 3. `split_block allocate_from_block(block_type const& b, std::size_t size)`
 * 4. `block_type free_block(void* p, std::size_t size) noexcept`
 */
template<typename PoolResource, typename FreeListType>
class stream_ordered_memory_resource: public crtp<PoolResource>, public device_memory_resource {
public:
  ~stream_ordered_memory_resource() override { release(); }

  stream_ordered_memory_resource()                                                 = default;
  stream_ordered_memory_resource(stream_ordered_memory_resource const&)            = delete;
  stream_ordered_memory_resource(stream_ordered_memory_resource&&)                 = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource const&) = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource&&)      = delete;

protected:
  using free_list  = FreeListType;
  using block_type = typename free_list::block_type;

  // Derived classes must implement these four methods

  /// Pair representing a block that has been split for allocation
  using split_block = std::pair<block_type, block_type>;

  /**
   * @brief Returns the block `b` (last used) to the pool.
   *
   * @param block The block to insert into the pool.
   */
  void insert_block(block_type const& block) {
    std::scoped_lock<std::mutex> lock(mutex_);
    this->free_blocks_.insert(block);
  }

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param size The size in bytes of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t size) override {
    if(size <= 0) { return nullptr; }

    auto const requested_size = size;
    size = rmm::detail::align_up(size, rmm::detail::RMM_ALLOCATION_ALIGNMENT);
    if(size < requested_size) {
      std::ostringstream os;
      os << "[TAMM ERROR] Allocation size overflowed while aligning request of "
         << requested_size << " bytes.\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    std::scoped_lock<std::mutex> lock(mutex_);

    if(!(size <= this->underlying().get_maximum_allocation_size())) {
      std::ostringstream os;
      os << "[TAMM ERROR] Maximum pool allocation size exceeded!\n" << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }
    auto const block = this->underlying().get_block(size);
#if TAMM_ENABLE_POOL_SANITY_CHECKS
    auto const insertion = live_blocks_.emplace(block.pointer(), block.size());
    if(!insertion.second) {
      std::ostringstream os;
      os << "[TAMM ERROR] Attempted to track duplicate allocation at " << block.pointer()
         << ".\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }
#endif
    return block.pointer();
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   * @param size The size in bytes of the allocation to deallocate
   */
  void do_deallocate(void* ptr, std::size_t size) override {
    if(size <= 0 || ptr == nullptr) { return; }

    auto const requested_size = size;
    size = rmm::detail::align_up(size, rmm::detail::RMM_ALLOCATION_ALIGNMENT);
    if(size < requested_size) {
      std::ostringstream os;
      os << "[TAMM ERROR] Deallocation size overflowed while aligning request of "
         << requested_size << " bytes.\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    std::scoped_lock<std::mutex> lock(mutex_);

#if TAMM_ENABLE_POOL_SANITY_CHECKS
    auto const iter = live_blocks_.find(ptr);
    if(iter == live_blocks_.end()) {
      std::ostringstream os;
      os << "[TAMM ERROR] Attempted to free pointer " << ptr
         << " that is not tracked by the pool resource.\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    if(iter->second != size) {
      std::ostringstream os;
      os << "[TAMM ERROR] Size mismatch while freeing pointer " << ptr
         << ": expected " << iter->second << " bytes but received " << size
         << " after alignment to RMM_ALLOCATION_ALIGNMENT ("
         << rmm::detail::RMM_ALLOCATION_ALIGNMENT << ").\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }

    live_blocks_.erase(iter);
#endif

    auto const block = this->underlying().free_block(ptr, size);
    free_blocks_.insert(block);
  }

private:
  /**
   * @brief Splits a block into an allocated block of `size` bytes and a remainder block, and
   * inserts the remainder into a free list.
   *
   * @param block The block to split into allocated and remainder portions.
   * @param size The size of the block to allocate from `b`.
   * @return The allocated block.
   */
  block_type allocate_and_insert_remainder(block_type block, std::size_t size) {
    auto const [allocated, remainder] = this->underlying().allocate_from_block(block, size);
    if(remainder.is_valid()) { free_blocks_.insert(remainder); }
    return allocated;
  }

  /**
   * @brief Get an available memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @return block_type A block of memory of at least `size` bytes
   */
  block_type get_block(std::size_t size) {
    block_type const block = free_blocks_.get_block(size);
    if(block.is_valid()) { return allocate_and_insert_remainder(block, size); }

    std::ostringstream os;
    os << "[TAMM ERROR] No memory-block found in stream_ordered_memory_resource!\n"
       << __FILE__ << ":L" << __LINE__;
    tamm_terminate(os.str());
    __builtin_unreachable();
  }

  /**
   * @brief Clear free lists
   *
   * Note: only called by destructor.
   */
  void release() {
    std::scoped_lock<std::mutex> lock(mutex_);
#if TAMM_ENABLE_POOL_SANITY_CHECKS
    if(!live_blocks_.empty()) {
      std::cerr << "[TAMM WARNING] Releasing pool memory resource with " << live_blocks_.size()
                << " outstanding allocations." << std::endl;
      live_blocks_.clear();
    }
#endif
    free_blocks_.clear();
  }

  free_list                              free_blocks_;
#if TAMM_ENABLE_POOL_SANITY_CHECKS
  std::unordered_map<void*, std::size_t> live_blocks_{};
#endif
  mutable std::mutex                     mutex_{};
}; // namespace detail

} // namespace tamm::rmm::mr::detail
