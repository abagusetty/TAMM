#pragma once

#include "device_memory_resource.hpp"
#include "gpu_memory_resource.hpp"

#if(defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define RMM_EXPORT __attribute__((visibility("default")))
#define RMM_HIDDEN __attribute__((visibility("hidden")))
#else
#define RMM_EXPORT
#define RMM_HIDDEN
#endif

#include <map>
#include <shared_mutex>

namespace tamm::rmm::mr {

namespace detail {

inline device_memory_resource* initial_resource() {
  static gpu_memory_resource mr{};
  return &mr;
}

/**
 * @brief Global device-id → resource map.
 *
 * Bug fixed vs. prior implementation:
 *   The map was read and written from multiple threads (MPI init, GPU pool
 *   init) without any synchronisation, constituting a data race (UB).
 *   A std::shared_mutex now allows concurrent reads while serialising writes.
 */
RMM_EXPORT inline auto& get_map() {
  static std::map<int, device_memory_resource*> device_id_to_resource;
  return device_id_to_resource;
}

RMM_EXPORT inline std::shared_mutex& get_map_mutex() {
  static std::shared_mutex mtx;
  return mtx;
}

} // namespace detail

/**
 * @brief Return the device_memory_resource registered for `device_id`.
 *
 * Thread-safe: uses a shared_lock for reads (common case) and upgrades to a
 * unique_lock only when inserting the default entry for an unseen device id.
 */
inline device_memory_resource* get_per_device_resource(int device_id) {
  auto& map = detail::get_map();
  auto& mtx = detail::get_map_mutex();

  // Fast path: shared (read) lock
  {
    std::shared_lock read_lock{mtx};
    auto it = map.find(device_id);
    if(it != map.end()) return it->second;
  }

  // Slow path: exclusive (write) lock for first-time insertion
  std::unique_lock write_lock{mtx};
  // Re-check after acquiring write lock (another thread may have inserted)
  auto it = map.find(device_id);
  if(it != map.end()) return it->second;
  return map.emplace(device_id, detail::initial_resource()).first->second;
}

} // namespace tamm::rmm::mr
