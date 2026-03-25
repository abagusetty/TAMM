#pragma once

#include "aligned.hpp"
#include "device_memory_resource.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>

namespace tamm::rmm::mr::detail {

template<typename T>
struct crtp {
  [[nodiscard]] T&       underlying()       { return static_cast<T&>(*this); }
  [[nodiscard]] T const& underlying() const { return static_cast<T const&>(*this); }
};

/**
 * @brief CRTP base providing stream-ordered suballocation logic.
 *
 * Changes vs. prior implementation:
 *
 * - The redundant second align_up inside do_allocate / do_deallocate is
 *   removed.  device_memory_resource::allocate() / deallocate() (the public
 *   non-virtual entry points) already align the size before dispatching to
 *   do_allocate / do_deallocate, so aligning again here was harmless but
 *   confusing and inconsistent with the CPU (host_memory_resource) path which
 *   does NOT pre-align in its public entry points.
 *
 * - A std::mutex member (mutex_) is added and exposed via get_mutex() so
 *   that pool_memory_resource::release() can lock it before iterating
 *   upstream_blocks_.  Without this lock, concurrent do_deallocate() calls
 *   during reset_rmm_pool() could insert into free_blocks_ after it has been
 *   cleared, leaking those blocks permanently.
 */
template<typename PoolResource, typename FreeListType>
class stream_ordered_memory_resource : public crtp<PoolResource>,
                                       public device_memory_resource {
public:
  ~stream_ordered_memory_resource() override { release(); }

  stream_ordered_memory_resource()                                                 = default;
  stream_ordered_memory_resource(stream_ordered_memory_resource const&)            = delete;
  stream_ordered_memory_resource(stream_ordered_memory_resource&&)                 = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource const&) = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource&&)      = delete;

  /// Exposed for derived pool_memory_resource::release() to lock before
  /// touching upstream_blocks_ / free_blocks_ concurrently.
  [[nodiscard]] std::mutex& get_mutex() noexcept { return mutex_; }

protected:
  using free_list  = FreeListType;
  using block_type = typename free_list::block_type;
  using split_block = std::pair<block_type, block_type>;

  void insert_block(block_type const& block) { free_blocks_.insert(block); }

  /**
   * @brief Allocate size bytes from the pool.
   *
   * NOTE: size arrives here already aligned by device_memory_resource::allocate().
   * We do NOT align again to avoid confusion; the assert documents the contract.
   */
  void* do_allocate(std::size_t size) override {
    if(size <= 0) return nullptr;

    // size must already be aligned by the public allocate() entry point
    assert(rmm::detail::is_aligned(size, rmm::detail::RMM_ALLOCATION_ALIGNMENT)
           && "do_allocate received unaligned size — public allocate() must align first");

    if(!(size <= this->underlying().get_maximum_allocation_size())) {
      std::ostringstream os;
      os << "[TAMM ERROR] Maximum pool allocation size exceeded!\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }
    auto const block = this->underlying().get_block(size);
    return block.pointer();
  }

  /**
   * @brief Return size bytes at ptr back to the pool.
   *
   * NOTE: size arrives here already aligned by device_memory_resource::deallocate().
   */
  void do_deallocate(void* ptr, std::size_t size) override {
    if(size <= 0 || ptr == nullptr) return;

    assert(rmm::detail::is_aligned(size, rmm::detail::RMM_ALLOCATION_ALIGNMENT)
           && "do_deallocate received unaligned size — public deallocate() must align first");

    auto const block = this->underlying().free_block(ptr, size);
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.insert(block);
  }

private:
  block_type allocate_and_insert_remainder(block_type block, std::size_t size) {
    auto const [allocated, remainder] = this->underlying().allocate_from_block(block, size);
    if(remainder.is_valid()) { free_blocks_.insert(remainder); }
    return allocated;
  }

  block_type get_block(std::size_t size) {
    block_type const block = free_blocks_.get_block(size);
    if(block.is_valid()) { return allocate_and_insert_remainder(block, size); }

    std::ostringstream os;
    os << "[TAMM ERROR] No memory-block found in stream_ordered_memory_resource!\n"
       << __FILE__ << ":L" << __LINE__;
    tamm_terminate(os.str());
    __builtin_unreachable();
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.clear();
  }

  free_list  free_blocks_;
  std::mutex mutex_;
};

} // namespace tamm::rmm::mr::detail
