#pragma once

#include <algorithm>
#include <concepts>
#include <iostream>
#include <set>

namespace tamm::rmm::mr::detail {

struct block_base {
  void* ptr{};

  block_base() = default;
  explicit block_base(void* p) noexcept : ptr{p} {}

  [[nodiscard]] void* pointer() const noexcept { return ptr; }
  [[nodiscard]] bool  is_valid() const noexcept { return ptr != nullptr; }
};

// C++20 concept constraining block types used with free_list
template<typename T>
concept BlockConcept = requires(T b, std::size_t n, T other) {
  { b.pointer()              } -> std::convertible_to<char*>;
  { b.size()                 } -> std::convertible_to<std::size_t>;
  { b.is_head()              } -> std::convertible_to<bool>;
  { b.is_valid()             } -> std::convertible_to<bool>;
  { b.fits(n)                } -> std::convertible_to<bool>;
  { b.is_contiguous_before(other) } -> std::convertible_to<bool>;
};

/**
 * @brief Comparator for block types ordered by pointer address.
 *
 * is_transparent enables heterogeneous lookup (find by raw char* without
 * constructing a temporary block).
 */
template<typename BlockType>
struct compare_blocks {
  using is_transparent = void;
  bool operator()(BlockType const& a, BlockType const& b) const noexcept {
    return a.pointer() < b.pointer();
  }
  bool operator()(char const* p, BlockType const& b) const noexcept {
    return p < b.pointer();
  }
  bool operator()(BlockType const& a, char const* p) const noexcept {
    return a.pointer() < p;
  }
};

/**
 * @brief Base class for a list of free memory blocks backed by std::set.
 *
 * Using std::set (ordered by address via compare_blocks) instead of std::list
 * enables O(log n) insert, erase, and lower_bound lookups used by
 * coalescing_free_list, replacing the previous O(n) std::find_if scans.
 */
template<BlockConcept BlockType>
class free_list {
public:
  free_list()          = default;
  virtual ~free_list() = default;

  free_list(free_list const&)            = delete;
  free_list& operator=(free_list const&) = delete;
  free_list(free_list&&)                 = delete;
  free_list& operator=(free_list&&)      = delete;

  using block_type     = BlockType;
  using set_type       = std::set<BlockType, compare_blocks<BlockType>>;
  using size_type      = typename set_type::size_type;
  using iterator       = typename set_type::iterator;
  using const_iterator = typename set_type::const_iterator;

  [[nodiscard]] iterator       begin()   noexcept       { return blocks_.begin(); }
  [[nodiscard]] const_iterator begin()   const noexcept { return blocks_.begin(); }
  [[nodiscard]] const_iterator cbegin()  const noexcept { return blocks_.cbegin(); }
  [[nodiscard]] iterator       end()     noexcept       { return blocks_.end(); }
  [[nodiscard]] const_iterator end()     const noexcept { return blocks_.end(); }
  [[nodiscard]] const_iterator cend()    const noexcept { return blocks_.cend(); }
  [[nodiscard]] size_type      size()    const noexcept { return blocks_.size(); }
  [[nodiscard]] bool           is_empty()const noexcept { return blocks_.empty(); }

  void erase(const_iterator it) { blocks_.erase(it); }
  void clear() noexcept { blocks_.clear(); }

  /**
   * @brief Merge all blocks from another free_list into this one.
   *
   * Uses std::set::merge (C++17) which moves nodes without reallocation.
   * Replaces the former splice() which was std::list-specific.
   */
  void merge_from(free_list&& other) { blocks_.merge(other.blocks_); }

protected:
  void insert(const_iterator /*hint*/, block_type const& block) { blocks_.insert(block); }
  void push_back(block_type const& block)  { blocks_.insert(block); }
  void push_back(block_type&& block)       { blocks_.insert(std::move(block)); }

  set_type blocks_;
};

} // namespace tamm::rmm::mr::detail
