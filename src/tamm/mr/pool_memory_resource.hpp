#pragma once

#include "aligned.hpp"
#include "coalescing_free_list.hpp"
#include "device_memory_resource.hpp"
#include "stream_ordered_memory_resource.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tamm::rmm::mr {

/**
 * @brief A coalescing best-fit suballocator backed by a pool from an upstream
 *        memory_resource.
 *
 * Key correctness properties vs. prior implementation:
 *
 * 1. is_head tracking via is_head_map_
 *    free_block() previously used upstream_blocks_.find(ptr) to decide if a
 *    returning pointer was a slab head, but upstream_blocks_ only stores the
 *    original slab pointers. Sub-allocations whose address happened to collide
 *    with a slab head (i.e. the very first sub-alloc from each slab) were
 *    incorrectly marked is_head=true, permanently blocking coalescing of their
 *    right neighbour. is_head_map_ records the is_head flag at allocation time.
 *
 * 2. std::logic_error is now actually thrown (not just constructed) for null
 *    upstream pointer and misaligned pool size.
 *
 * 3. maximum_pool_size_ is properly stored and respected.
 *
 * 4. Upstream pointer ownership: upstream_mr_ is still owned (deleted in dtor)
 *    but release() is called before delete so all upstream deallocate() calls
 *    complete before the upstream resource is destroyed.
 */
template<typename Upstream>
class pool_memory_resource final
  : public detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                  detail::coalescing_free_list> {
public:
  friend class detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                      detail::coalescing_free_list>;

  explicit pool_memory_resource(Upstream* upstream_mr, std::size_t maximum_pool_size)
    : upstream_mr_{[upstream_mr]() {
        if(upstream_mr == nullptr)
          throw std::logic_error("pool_memory_resource: null upstream pointer");
        return upstream_mr;
      }()}
    , maximum_pool_size_{maximum_pool_size}
  {
    if(!rmm::detail::is_aligned(maximum_pool_size, rmm::detail::RMM_ALLOCATION_ALIGNMENT))
      throw std::logic_error(
        "pool_memory_resource: maximum_pool_size must be a multiple of "
        "RMM_ALLOCATION_ALIGNMENT bytes");

    initialize_pool(maximum_pool_size);
  }

  /**
   * @brief Destroy the pool: release all upstream slabs first, then delete
   *        the upstream resource. Order matters - upstream_mr_ must still be
   *        alive when its deallocate() is called inside release().
   */
  ~pool_memory_resource() override {
    release();
    delete upstream_mr_;
    upstream_mr_ = nullptr;
  }

  pool_memory_resource()                                       = delete;
  pool_memory_resource(pool_memory_resource const&)            = delete;
  pool_memory_resource(pool_memory_resource&&)                 = delete;
  pool_memory_resource& operator=(pool_memory_resource const&) = delete;
  pool_memory_resource& operator=(pool_memory_resource&&)      = delete;

  [[nodiscard]] Upstream*   get_upstream()         const noexcept { return upstream_mr_; }
  [[nodiscard]] std::size_t get_maximum_pool_size() const noexcept { return maximum_pool_size_; }

protected:
  using free_list  = detail::coalescing_free_list;
  using block_type = free_list::block_type;
  using typename detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                        detail::coalescing_free_list>::split_block;

  [[nodiscard]] std::size_t get_maximum_allocation_size() const noexcept {
    return std::numeric_limits<std::size_t>::max();
  }

  void initialize_pool(std::size_t maximum_size) {
    auto const block = block_from_upstream(maximum_size);
    if(block.has_value()) {
      this->insert_block(block.value());
    }
    else {
      std::ostringstream os;
      os << "[TAMM ERROR] RMM initialize_pool() failed, too many processes per node!\n"
         << __FILE__ << ":L" << __LINE__;
      tamm_terminate(os.str());
    }
  }

  /**
   * @brief Allocate a slab from upstream and register it in both tracking
   *        structures.
   */
  std::optional<block_type> block_from_upstream(std::size_t size) {
    if(size == 0) return std::nullopt;
    try {
      void* ptr  = get_upstream()->allocate(size);
      char* cptr = static_cast<char*>(ptr);
      auto [it, inserted] = upstream_blocks_.emplace(cptr, size, true);
      (void)inserted;
      is_head_map_.emplace(cptr, true);
      return std::optional<block_type>{*it};
    }
    catch(std::exception const&) { return std::nullopt; }
  }

  split_block allocate_from_block(block_type const& block, std::size_t size) {
    block_type const alloc{block.pointer(), size, block.is_head()};
    auto rest = (block.size() > size)
                  ? block_type{block.pointer() + size, block.size() - size, false}
                  : block_type{};
    return {alloc, rest};
  }

  /**
   * @brief Reconstruct the block descriptor for a pointer being freed.
   *
   * is_head is looked up from is_head_map_ (populated at slab-allocation
   * time) rather than inferred from upstream_blocks_.find(ptr). The old
   * approach gave false positives for the first sub-allocation of each slab
   * (same address as the slab head), permanently blocking coalescing.
   */
  block_type free_block(void* ptr, std::size_t size) noexcept {
    char* cptr      = static_cast<char*>(ptr);
    bool  head_flag = (is_head_map_.count(cptr) > 0);
    return block_type{cptr, size, head_flag};
  }

  /**
   * @brief Return all upstream slabs to the upstream resource.
   */
  void release() {
    for(auto const& blk : upstream_blocks_) {
      get_upstream()->deallocate(blk.pointer(), blk.size());
    }
    upstream_blocks_.clear();
    is_head_map_.clear();
  }

private:
  Upstream*   upstream_mr_;
  std::size_t maximum_pool_size_{};

  // Slab-level tracking: one entry per upstream allocate() call
  std::set<block_type, rmm::mr::detail::compare_blocks<block_type>> upstream_blocks_;

  // is_head flag keyed by slab base pointer - avoids false positives in free_block()
  std::unordered_map<char*, bool> is_head_map_;
};

} // namespace tamm::rmm::mr
