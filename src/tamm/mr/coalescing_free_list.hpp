#pragma once

#include "free_list.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <set>

namespace tamm::rmm::mr::detail {

/**
 * @brief A block of memory with pointer, size, and upstream-head flag.
 *
 * Blocks are ordered by pointer address. Two adjacent blocks where the
 * lower block's is_contiguous_before() returns true may be coalesced.
 */
struct block : public block_base {
  block() = default;
  block(char* p, std::size_t sz, bool head) noexcept
    : block_base{p}, size_bytes{sz}, head{head} {}

  [[nodiscard]] char*       pointer()  const noexcept { return static_cast<char*>(ptr); }
  [[nodiscard]] std::size_t size()     const noexcept { return size_bytes; }
  [[nodiscard]] bool        is_head()  const noexcept { return head; }

  bool operator<(block const& rhs) const noexcept { return pointer() < rhs.pointer(); }

  [[nodiscard]] bool fits(std::size_t bytes) const noexcept { return size_bytes >= bytes; }

  /**
   * @brief True if this block immediately precedes `b` AND `b` is not the
   * start of a separate upstream slab (b.is_head() == false).
   *
   * Note: we check b.is_head(), NOT this->is_head(). A head block CAN have
   * something coalesced before it only from the same upstream slab — this is
   * correctly prevented because head marks the START of an upstream slab, so
   * nothing legitimately precedes it within the same slab.
   */
  [[nodiscard]] bool is_contiguous_before(block const& b) const noexcept {
    return (pointer() + size_bytes == b.ptr) && !b.is_head();
  }

  [[nodiscard]] block merge(block const& b) const noexcept {
    assert(is_contiguous_before(b));
    return {pointer(), size_bytes + b.size_bytes, head};
  }

private:
  std::size_t size_bytes{};
  bool        head{false};
};

/**
 * @brief Comparator ordering blocks by (size, pointer) for the size index.
 *
 * Enables best-fit lookup: lower_bound on a sentinel block of exactly the
 * requested size finds the smallest block >= that size in O(log n).
 */
struct size_ptr_less {
  using is_transparent = void;
  bool operator()(block const& a, block const& b) const noexcept {
    if(a.size() != b.size()) return a.size() < b.size();
    return a.pointer() < b.pointer();
  }
  // Heterogeneous lookup by (size_t, char*) pair
  bool operator()(std::size_t sz, block const& b) const noexcept { return sz < b.size(); }
  bool operator()(block const& a, std::size_t sz) const noexcept { return a.size() < sz; }
};

/**
 * @brief An ordered free list that coalesces contiguous blocks on insertion.
 *
 * Maintains two indices:
 *   blocks_     — address-ordered std::set for O(log n) neighbour lookup
 *   size_index_ — (size, ptr)-ordered std::set for O(log n) best-fit search
 *
 * Both indices are kept in sync on every insert/erase.
 */
struct coalescing_free_list : free_list<block> {
  coalescing_free_list()           = default;
  ~coalescing_free_list() override = default;

  coalescing_free_list(coalescing_free_list const&)            = delete;
  coalescing_free_list& operator=(coalescing_free_list const&) = delete;
  coalescing_free_list(coalescing_free_list&&)                 = delete;
  coalescing_free_list& operator=(coalescing_free_list&&)      = delete;

  /**
   * @brief Insert a block, coalescing with address-adjacent neighbours.
   *
   * O(log n) — uses std::set::lower_bound to locate neighbours, then
   * updates both indices atomically.
   */
  void insert(block_type const& blk) {
    if(!blk.is_valid() || blk.size() == 0) return;

    // Locate the first block with address > blk (the potential right neighbour)
    auto next = blocks_.lower_bound(blk);

    bool        merge_next = (next != blocks_.end()) && blk.is_contiguous_before(*next);
    bool        merge_prev = false;
    iterator    prev;

    if(next != blocks_.begin()) {
      prev       = std::prev(next);
      merge_prev = prev->is_contiguous_before(blk);
    }

    block merged = blk;

    if(merge_prev && merge_next) {
      remove_from_size_index(*prev);
      remove_from_size_index(*next);
      merged = prev->merge(blk).merge(*next);
      auto next_it = next; // next iterator still valid before erasing prev
      blocks_.erase(prev);
      blocks_.erase(next_it);
    }
    else if(merge_prev) {
      remove_from_size_index(*prev);
      merged = prev->merge(blk);
      blocks_.erase(prev);
    }
    else if(merge_next) {
      remove_from_size_index(*next);
      merged = blk.merge(*next);
      blocks_.erase(next);
    }

    blocks_.insert(merged);
    size_index_.insert(merged);
  }

  /**
   * @brief Find and remove the smallest free block >= `size` bytes.
   *
   * O(log n) — uses lower_bound on the size-ordered secondary index.
   *
   * @param size Requested allocation size (must already be align_up'd).
   * @return Matching block, or default-constructed (invalid) block if none.
   */
  block_type get_block(std::size_t size) {
    // Sentinel: find first block whose size >= requested size
    auto it = size_index_.lower_bound(size);

    if(it == size_index_.end()) return block_type{}; // pool exhausted

    block found = *it;
    size_index_.erase(it);
    blocks_.erase(blocks_.find(found)); // O(log n) find in address set
    return found;
  }

  void clear() noexcept {
    free_list<block>::clear();
    size_index_.clear();
  }

private:
  void remove_from_size_index(block const& b) {
    auto it = size_index_.find(b);
    if(it != size_index_.end()) size_index_.erase(it);
  }

  // Secondary index: ordered by (size, pointer) for O(log n) best-fit lookup
  std::set<block, size_ptr_less> size_index_;
};

} // namespace tamm::rmm::mr::detail
