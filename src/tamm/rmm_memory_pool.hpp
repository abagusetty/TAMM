#pragma once

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#else
#include <ga/ga.h>
#endif

#include <tamm/gpu_streams.hpp>

#include "tamm/mr/gpu_memory_resource.hpp"
#include "tamm/mr/per_device_resource.hpp"
#include "tamm/mr/pinned_memory_resource.hpp"
#endif

#include <tamm/errors.hpp>

#include "tamm/mr/device_memory_resource.hpp"
#include "tamm/mr/host_memory_resource.hpp"
#include "tamm/mr/new_delete_resource.hpp"
#include "tamm/mr/pool_memory_resource.hpp"

namespace tamm {

namespace detail {
// TAMM_ENABLE_SPRHBM = 0(default), 1
static const uint32_t tamm_enable_sprhbm = [] {
  const char* tammEnableSprHBM = std::getenv("TAMM_ENABLE_SPRHBM");
  uint32_t    usinghbm         = 0;
  if(tammEnableSprHBM != nullptr) { usinghbm = std::atoi(tammEnableSprHBM); }
  return usinghbm;
}();

// TAMM_GPU_POOL
static const uint32_t tamm_gpu_pool = [] {
  uint32_t usinggpupool = 80;
  if(const char* tammGpupoolsize = std::getenv("TAMM_GPU_POOL")) {
    usinggpupool = std::atoi(tammGpupoolsize);
  }
  return usinggpupool;
}();

// TAMM_CPU_POOL
static const uint32_t tamm_cpu_pool = [] {
  uint32_t usingcpupool = 100;
  if(const char* tammCpupoolsize = std::getenv("TAMM_CPU_POOL")) {
    usingcpupool = std::atoi(tammCpupoolsize);
  }
  return usingcpupool;
}();

// TAMM_RANKS_PER_GPU_POOL
static const uint32_t tamm_rpg = [] {
  uint32_t usingrpg = 1;
  if(const char* tammrpg = std::getenv("TAMM_RANKS_PER_GPU_POOL")) {
    usingrpg = std::atoi(tammrpg);
  }
  return usingrpg;
}();
} // namespace detail

class RMMMemoryManager {
protected:
  bool invalid_state{true};
  using host_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::host_memory_resource>;
  std::unique_ptr<host_pool_mr> hostMR;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  using device_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  std::unique_ptr<device_pool_mr> deviceMR;
#endif
  // #if defined(USE_CUDA) || defined(USE_HIP)
  //   using pinned_pool_mr = rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>;
  //   std::unique_ptr<pinned_pool_mr> pinnedHostMR;
  // #endif

private:
  RMMMemoryManager() { initialize(); }

public:
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  /// Returns a RMM device pool handle
  device_pool_mr& getDeviceMemoryPool() { return *(deviceMR.get()); }
#endif

  // #if defined(USE_CUDA) || defined(USE_HIP)
  //   /// Returns a RMM pinnedHost pool handle
  //   pinned_pool_mr& getPinnedMemoryPool() { return *(pinnedHostMR.get()); }
  // #elif defined(USE_DPCPP)
  //   /// Returns a RMM pinnedHost pool handle
  //   host_pool_mr& getPinnedMemoryPool() { return *(hostMR.get()); }
  // #endif

  /// Returns a RMM host pool handle
  host_pool_mr& getHostMemoryPool() { return *(hostMR.get()); }

  /// Returns the instance of device manager singleton.
  inline static RMMMemoryManager& getInstance() {
    static RMMMemoryManager d_m{};
    return d_m;
  }

  void reset() {
    hostMR.reset();
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    deviceMR.reset();
// #elif defined(USE_CUDA) || defined(USE_HIP)
//     pinnedHostMR.reset();
#endif

    this->invalid_state = true;
  }

  void initialize() {
    if(this->invalid_state) {
      // Number of user-MPI ranks is needed for efficient CPU-pool size
      int ranks_pn_ = 0;
#if defined(USE_UPCXX)
      ranks_pn_ = upcxx::local_team().rank_n();
#else
      ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_nodeid());
#endif

      long max_host_bytes{0};

#if defined(__APPLE__)
      size_t cpu_mem_per_node;
      size_t size_mpn = sizeof(cpu_mem_per_node);
      // TODO: query for freeram, not total
      sysctlbyname("hw.memsize", &(cpu_mem_per_node), &size_mpn, nullptr, 0);
      max_host_bytes = 0.5 * cpu_mem_per_node;
      // Use only "tamm_cpu_pool" percent of the remaining memory
      max_host_bytes *= (detail::tamm_cpu_pool / 100.0);
#else
      // Set the CPU memory-pool
      EXPECTS_STR((numa_available() != -1), "[TAMM ERROR]: numa APIs are not available!");

      numa_set_bind_policy(1);
      unsigned numNumaNodes = numa_num_task_nodes();

      // for ranks_pn_=1, there is no need to check the mapping to numa-nodes (mostly used for CI)
      // for ranks_pn_ > numNumaNodes, it has to be divisble by the number of numa-domains in the
      // system
      if(ranks_pn_ >= numNumaNodes && ranks_pn_ > 1) {
        EXPECTS_STR((ranks_pn_ % numNumaNodes == 0),
                    "[TAMM ERROR]: number of user ranks is not a multiple of numa-nodes!");
      }
      struct bitmask* numaNodes = numa_get_mems_allowed();
      numa_bind(numaNodes);

      int  numa_id         = numa_preferred();
      long numa_total_size = numa_node_size(numa_id, &max_host_bytes);
      max_host_bytes *= 0.40; // reserve 40% only of the free numa-node memory (reserving rest of
                              // GA, non-pool allocations)

      if(numNumaNodes > 1) { // please the systems with just 1 Numa partitions
        // Identify the NUMA distance for faster numa-regions
        std::map<int, int> numadist_id;
        for(int j = 0; j < numNumaNodes; j++) {
          if(numa_id != j) { numadist_id[j] = numa_distance(numa_id, j); }
        }
        int  val    = numadist_id.begin()->second;
        auto result = std::all_of(
          std::next(numadist_id.begin()), numadist_id.end(),
          [val](typename std::map<int, int>::const_reference t) { return t.second == val; });
        if(!result) { // There are some faster NUMA domains available than the defaults (only for
                      // Aurora)
          auto it =
            std::min_element(numadist_id.begin(), numadist_id.end(),
                             [](const auto& l, const auto& r) { return l.second < r.second; });

          numNumaNodes /= 2; // This is done for the Aurora nodes only

          if(detail::tamm_enable_sprhbm) {
            numa_id = it->first;
            numa_set_preferred(numa_id);
            numa_total_size = numa_node_size(numa_id, &max_host_bytes);
            max_host_bytes *=
              0.94; // One can use full HBM memory capacity, since the DDR is left for GA
          }
        }
      } // numNumaNodes > 1

      max_host_bytes *=
        (detail::tamm_cpu_pool / 100.0); // Use only "tamm_cpu_pool" percent of the left-overs
      max_host_bytes /= ((numNumaNodes > 1)
                           ? ((ranks_pn_ >= numNumaNodes) ? (ranks_pn_ / numNumaNodes) : 1)
                           : ranks_pn_);
#endif

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
      size_t free{}, total{};
      gpuMemGetInfo(&free, &total);
      size_t max_device_bytes{0};
      max_device_bytes = ((detail::tamm_gpu_pool / 100.0) * free) / detail::tamm_rpg;

      deviceMR =
        std::make_unique<device_pool_mr>(new rmm::mr::gpu_memory_resource, max_device_bytes);
      // #if defined(USE_CUDA) || defined(USE_HIP)
      //       size_t max_pinned_host_bytes{0};
      //       max_pinned_host_bytes = 0.18 * free;
      //       pinnedHostMR = std::make_unique<pinned_pool_mr>(new rmm::mr::pinned_memory_resource,
      //                                                       max_pinned_host_bytes);
      // #endif
#endif
      hostMR = std::make_unique<host_pool_mr>(new rmm::mr::new_delete_resource, max_host_bytes);

      // after setting up the pool: change the invalid_state to FALSE
      invalid_state = false;
    }
  }

  RMMMemoryManager(const RMMMemoryManager&)            = delete;
  RMMMemoryManager& operator=(const RMMMemoryManager&) = delete;
  RMMMemoryManager(RMMMemoryManager&&)                 = delete;
  RMMMemoryManager& operator=(RMMMemoryManager&&)      = delete;
};

// The reset pool & reinitialize only is being used for the (T) segement of cannonical
static inline void reset_rmm_pool() { RMMMemoryManager::getInstance().reset(); }

static inline void reinitialize_rmm_pool() { RMMMemoryManager::getInstance().initialize(); }

} // namespace tamm
