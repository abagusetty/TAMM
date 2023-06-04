#include "ga/ga.h"
#include <mpi.h>

#include "distribution.hpp"
#include "execution_context.hpp"
#include "labeled_tensor.hpp"
#include "memory_manager.hpp"
#include "proc_group.hpp"
#include "runtime_engine.hpp"

#if __APPLE__
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif

namespace tamm {
ExecutionContext::ExecutionContext(ProcGroup pg, DistributionKind default_dist_kind,
                                   MemoryManagerKind default_memory_manager_kind,
                                   RuntimeEngine*    re):
  pg_{pg},
  distribution_kind_{default_dist_kind},
  memory_manager_kind_{default_memory_manager_kind},
  ac_{IndexedAC{nullptr, 0}} {
  if(re == nullptr) { re_.reset(runtime_ptr()); }
  else {
    re_.reset(re, [](auto) {});
  }

#if defined(USE_UPCXX)
  pg_self_ = ProcGroup{team_self};

#if defined(USE_UPCXX_DISTARRAY)
  hint_ = pg.size().value();
#endif

#else
  pg_self_  = ProcGroup{MPI_COMM_SELF, ProcGroup::self_ga_pgroup()};
#endif

#if defined(USE_UPCXX)
  ranks_pn_ = upcxx::local_team().rank_n();
#else
  ranks_pn_ = GA_Cluster_nprocs(GA_Cluster_proc_nodeid(pg.rank().value()));
#endif
  nnodes_ = pg.size().value() / ranks_pn_;

#if __APPLE__
  {
    size_t size_mpn = sizeof(minfo_.cpu_mem_per_node);
    sysctlbyname("hw.memsize", &(minfo_.cpu_mem_per_node), &size_mpn, nullptr, 0);
  }
#else
  {
    struct sysinfo cpumeminfo_;
    sysinfo(&cpumeminfo_);
    minfo_.cpu_mem_per_node = cpumeminfo_.totalram * cpumeminfo_.mem_unit;
  }
#endif
  minfo_.cpu_mem_per_node /= (1024 * 1024 * 1024.0); // GiB
  minfo_.total_cpu_mem = minfo_.cpu_mem_per_node * nnodes_;

#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  int ngpu_{0};
  tamm::getDeviceCount(&ngpu_);
  int dev_id_ = ((pg.rank().value() % ranks_pn_) % ngpu_);
  if(ngpu_ == 1) dev_id_ = 0;

  has_gpu_ = true;
  exhw_    = ExecutionHW::GPU;

  {
    size_t free_{};
    gpuMemGetInfo(&free_, &minfo_.gpu_mem_per_device);
    minfo_.gpu_mem_per_device /= (1024 * 1024 * 1024.0); // GiB
  }
#endif

  // GPUStreamPool as singleton object
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
  if(ngpu_ > 1) {
    auto& pool = tamm::GPUStreamPool::getInstance();
    pool.set_device(dev_id_);
  }
#endif
}

ExecutionContext::ExecutionContext(ProcGroup pg, Distribution* default_distribution,
                                   MemoryManager* default_memory_manager, RuntimeEngine* re):
  ExecutionContext{
    pg, default_distribution != nullptr ? default_distribution->kind() : DistributionKind::invalid,
    default_memory_manager != nullptr ? default_memory_manager->kind() : MemoryManagerKind::invalid,
    re} {}

void ExecutionContext::set_distribution(Distribution* distribution) {
  if(distribution) { distribution_kind_ = distribution->kind(); }
  else { distribution_kind_ = DistributionKind::invalid; }
}

void ExecutionContext::set_re(RuntimeEngine* re) { re_.reset(re); }

} // namespace tamm
