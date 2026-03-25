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

namespace tamm::rmm::mr {

namespace detail {

inline device_memory_resource* initial_resource() {
  static gpu_memory_resource mr{};
  return &mr;
}

// Must have default visibility, see: https://github.com/rapidsai/rmm/issues/826
RMM_EXPORT inline auto& get_map() {
  static std::map<int, device_memory_resource*> device_id_to_resource;
  return device_id_to_resource;
}

} // namespace detail

/**
 * @brief Get the resource for the specified device.
 *
 * Returns a pointer to the device_memory_resource for the specified device.
 * The initial resource is a gpu_memory_resource.
 *
 * @param device_id The id of the target device
 * @return Pointer to the current device_memory_resource for device_id
 */
inline device_memory_resource* get_per_device_resource(int device_id) {
  auto& map   = detail::get_map();
  auto  found = map.find(device_id);
  return (found == map.end()) ? (map[device_id] = detail::initial_resource()) : found->second;
}

} // namespace tamm::rmm::mr
