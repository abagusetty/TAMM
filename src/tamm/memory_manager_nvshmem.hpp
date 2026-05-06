#pragma once

/**
 * @file memory_manager_nvshmem.hpp
 * @brief GPU-resident distributed memory manager for TAMM.
 *
 * Uses the NVSHMEM symmetric heap for collective GPU allocation and
 * GPU-aware MPI one-sided RMA (MPI_Win over CUDA device buffers) for ALL
 * inter-rank data movement — get, put, and accumulate.
 *
 * Design principles:
 *   - NVSHMEM is used ONLY for:
 *       * nvshmem_malloc / nvshmem_free  (symmetric GPU heap)
 *       * nvshmem_size_g gather          (exchange per-PE sizes at alloc)
 *       * nvshmem_barrier_all            (collective sync)
 *   - ALL remote communication (local or remote node) uses:
 *       MPI_Get / MPI_Put / MPI_Accumulate(MPI_SUM)
 *       over a CUDA-aware MPI_Win created on the GPU buffer.
 *   - There is NO UPC++ dependency anywhere in this file.
 *   - A CUDA kernel is used only for same-rank self-add (no MPI round-trip).
 *
 * Prerequisites:
 *   - nvshmemx_init_attr() called before TAMM init (e.g. in tamm.cpp)
 *   - CUDA-aware MPI (OpenMPI --with-cuda, Cray MPICH, or MVAPICH2-GDR)
 *   - Compile with -DUSE_NVSHMEM
 *
 * Usage:
 *   auto* mm = MemoryManagerNVSHMEM::create_coll(pg);
 *   // ... tensor operations ...
 *   MemoryManagerNVSHMEM::destroy_coll(mm);
 */

#ifdef USE_NVSHMEM

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>

#include <complex>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "tamm/errors.hpp"
#include "tamm/memory_manager.hpp"
#include "tamm/proc_group.hpp"
#include "tamm/types.hpp"

// ---------------------------------------------------------------------------
// CUDA kernel: self-add only (same rank, no MPI needed)
// ---------------------------------------------------------------------------
namespace tamm_nvshmem_kernels {

template<typename T>
__global__ void accumulate_kernel(T* __restrict__ dst, const T* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) dst[i] += src[i];
}

__global__ void accumulate_kernel_cf(float2* __restrict__  dst,
                                      const float2* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) { dst[i].x += src[i].x; dst[i].y += src[i].y; }
}

__global__ void accumulate_kernel_cd(double2* __restrict__ dst,
                                      const double2* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) { dst[i].x += src[i].x; dst[i].y += src[i].y; }
}

inline void launch_self_accumulate(void* dst, const void* src,
                                   ElementType eltype, size_t n, cudaStream_t stream) {
  const int threads = 256;
  const size_t blocks = (n + threads - 1) / threads;
  switch(eltype) {
    case ElementType::single_precision:
      accumulate_kernel<float><<<blocks,threads,0,stream>>>(
        reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(src), n);
      break;
    case ElementType::double_precision:
      accumulate_kernel<double><<<blocks,threads,0,stream>>>(
        reinterpret_cast<double*>(dst), reinterpret_cast<const double*>(src), n);
      break;
    case ElementType::single_complex:
      accumulate_kernel_cf<<<blocks,threads,0,stream>>>(
        reinterpret_cast<float2*>(dst), reinterpret_cast<const float2*>(src), n);
      break;
    case ElementType::double_complex:
      accumulate_kernel_cd<<<blocks,threads,0,stream>>>(
        reinterpret_cast<double2*>(dst), reinterpret_cast<const double2*>(src), n);
      break;
    default: UNREACHABLE();
  }
}

} // namespace tamm_nvshmem_kernels

// ---------------------------------------------------------------------------

namespace tamm {

class MemoryManagerNVSHMEM;

// ---------------------------------------------------------------------------
// MemoryRegionNVSHMEM
// ---------------------------------------------------------------------------

/**
 * @ingroup memory_management
 * @brief Memory region backed by the NVSHMEM symmetric GPU heap.
 *
 * Collective allocation distributes tensor blocks across all PEs. A per-PE
 * byte-offset table allows any rank to compute the MPI displacement for any
 * remote element without extra communication.
 *
 * An MPI_Win is created over each rank's local GPU slice. CUDA-aware MPI
 * handles all data movement — the NIC reads/writes GPU memory directly.
 */
class MemoryRegionNVSHMEM: public MemoryRegionImpl<MemoryManagerNVSHMEM> {
public:
  explicit MemoryRegionNVSHMEM(MemoryManagerNVSHMEM& mgr)
      : MemoryRegionImpl<MemoryManagerNVSHMEM>(mgr) {}

private:
  // Symmetric GPU heap pointer (nvshmem_malloc'd, same logical address on all PEs)
  void*  symm_base_{nullptr};
  size_t symm_total_bytes_{0};

  // Per-PE byte offsets into symm_base_ (exclusive prefix sum of per-PE sizes)
  // pe_byte_offsets_[i] = byte start of PE i's local slice
  std::vector<size_t> pe_byte_offsets_;

  ElementType eltype_{ElementType::invalid};
  size_t      elsize_{0};

  // GPU-aware MPI window created over this rank's local GPU slice.
  // All inter-rank get/put/accumulate operations go through this window.
  MPI_Win mpi_win_{MPI_WIN_NULL};

  friend class MemoryManagerNVSHMEM;
};

// ---------------------------------------------------------------------------
// MemoryManagerNVSHMEM
// ---------------------------------------------------------------------------

/**
 * @ingroup memory_management
 * @brief Distributed memory manager: NVSHMEM heap + GPU-aware MPI one-sided RMA.
 *
 * Drop-in replacement for MemoryManagerGA for GPU-resident tensor storage.
 * All inter-rank communication uses MPI_Get / MPI_Put / MPI_Accumulate over
 * a CUDA-aware MPI_Win — no UPC++, no ARMCI, no GA remote ops.
 *
 * Enable at compile time:  -DUSE_NVSHMEM
 * Select at runtime via ExecutionContext with MemoryManagerKind::nvshmem.
 */
class MemoryManagerNVSHMEM: public MemoryManager {
public:

  // -- Factory --------------------------------------------------------------

  static MemoryManagerNVSHMEM* create_coll(ProcGroup pg) {
    return new MemoryManagerNVSHMEM{pg};
  }

  static void destroy_coll(MemoryManagerNVSHMEM* mm) { delete mm; }

  // -- alloc_coll -----------------------------------------------------------

  /**
   * Collective allocation supporting irregular layouts (different nelements per PE).
   *
   * Steps:
   *  1. Exchange per-PE byte sizes via NVSHMEM symmetric scratch + nvshmem_size_g.
   *  2. Build exclusive prefix-sum offset table (pe_byte_offsets_).
   *  3. Allocate a single contiguous symmetric GPU buffer (nvshmem_malloc).
   *  4. Create a CUDA-aware MPI_Win over this rank's local slice.
   *  5. Open a persistent MPI_Win epoch (MPI_Win_lock_all).
   */
  MemoryRegion* alloc_coll(ElementType eltype, Size local_nelements) override {
    auto* pmr          = new MemoryRegionNVSHMEM(*this);
    pmr->eltype_       = eltype;
    pmr->elsize_       = element_size(eltype);
    pmr->local_nelements_ = local_nelements;

    const int    npes        = pg_.size().value();
    const int    my_pe       = pg_.rank().value();
    const size_t local_bytes = local_nelements.value() * pmr->elsize_;

    // 1. Exchange sizes: each PE writes its own local_bytes into a symmetric
    //    scratch array, then gathers all entries via nvshmem_size_g.
    size_t* symm_scratch = static_cast<size_t*>(nvshmem_malloc(sizeof(size_t) * npes));
    EXPECTS(symm_scratch != nullptr);
    symm_scratch[my_pe] = local_bytes;
    nvshmem_barrier_all();

    std::vector<size_t> pe_sizes(npes);
    for(int pe = 0; pe < npes; ++pe) {
      pe_sizes[pe] = nvshmem_size_g(&symm_scratch[pe], pe);
    }
    nvshmem_free(symm_scratch);

    // 2. Exclusive prefix sum -> byte offsets
    pmr->pe_byte_offsets_.resize(npes + 1);
    pmr->pe_byte_offsets_[0] = 0;
    for(int pe = 0; pe < npes; ++pe) {
      pmr->pe_byte_offsets_[pe + 1] = pmr->pe_byte_offsets_[pe] + pe_sizes[pe];
    }
    pmr->symm_total_bytes_ = pmr->pe_byte_offsets_[npes];

    // 3. Allocate contiguous symmetric GPU buffer
    pmr->symm_base_ = nvshmem_malloc(pmr->symm_total_bytes_);
    EXPECTS(pmr->symm_base_ != nullptr);
    if(local_bytes > 0) {
      cudaMemset(
        static_cast<uint8_t*>(pmr->symm_base_) + pmr->pe_byte_offsets_[my_pe],
        0, local_bytes);
    }

    // 4. Create GPU-aware MPI_Win over local slice
    //    The window base is a GPU device pointer; CUDA-aware MPI passes it
    //    directly to the NIC without host staging.
    MPI_Win_create(
      static_cast<uint8_t*>(pmr->symm_base_) + pmr->pe_byte_offsets_[my_pe],
      static_cast<MPI_Aint>(local_bytes),
      /*disp_unit=*/1,
      MPI_INFO_NULL,
      pg_.comm(),
      &pmr->mpi_win_);

    // 5. Open a persistent passive-target epoch (matches GA's always-open semantics)
    MPI_Win_lock_all(MPI_MODE_NOCHECK, pmr->mpi_win_);

    nvshmem_barrier_all();
    pmr->set_status(AllocationStatus::created);
    return pmr;
  }

  /**
   * Balanced allocation: all ranks contribute max_nelements.
   * Delegates to alloc_coll.
   */
  MemoryRegion* alloc_coll_balanced(ElementType eltype, Size max_nelements,
                                    ProcList proc_list = {}) override {
    EXPECTS(proc_list.empty());
    return alloc_coll(eltype, max_nelements);
  }

  // -- attach_coll ----------------------------------------------------------

  /**
   * Attach creates a second handle sharing the existing GPU allocation.
   * The underlying nvshmem buffer and MPI_Win are NOT duplicated.
   */
  MemoryRegion* attach_coll(MemoryRegion& mrb) override {
    auto& src = static_cast<MemoryRegionNVSHMEM&>(mrb);
    auto* pmr = new MemoryRegionNVSHMEM(*this);

    pmr->symm_base_        = src.symm_base_;
    pmr->symm_total_bytes_ = src.symm_total_bytes_;
    pmr->eltype_           = src.eltype_;
    pmr->elsize_           = src.elsize_;
    pmr->pe_byte_offsets_  = src.pe_byte_offsets_;
    pmr->local_nelements_  = src.local_nelements_;
    pmr->mpi_win_          = src.mpi_win_; // shared, not owned

    pmr->set_status(AllocationStatus::attached);
    pg_.barrier();
    return pmr;
  }

  // -- dealloc / detach -----------------------------------------------------

  void dealloc_coll(MemoryRegion& mrb) override {
    auto& mr = static_cast<MemoryRegionNVSHMEM&>(mrb);
    nvshmem_barrier_all();
    if(mr.mpi_win_ != MPI_WIN_NULL) {
      MPI_Win_unlock_all(mr.mpi_win_);
      MPI_Win_free(&mr.mpi_win_);
      mr.mpi_win_ = MPI_WIN_NULL;
    }
    if(mr.symm_base_) {
      nvshmem_free(mr.symm_base_);
      mr.symm_base_ = nullptr;
    }
  }

  void detach_coll(MemoryRegion& mrb) override {
    auto& mr      = static_cast<MemoryRegionNVSHMEM&>(mrb);
    mr.symm_base_ = nullptr;
    mr.mpi_win_   = MPI_WIN_NULL;
    pg_.barrier();
  }

  // -- fence ----------------------------------------------------------------

  /**
   * Drains all pending GPU-aware MPI RMA operations on this window.
   * Must be called before reading data written by a remote put/accumulate.
   */
  void fence(MemoryRegion& mrb) override {
    auto& mr = static_cast<MemoryRegionNVSHMEM&>(mrb);
    if(mr.mpi_win_ != MPI_WIN_NULL) {
      MPI_Win_flush_all(mr.mpi_win_);
    }
  }

  // -- access (local pointer) -----------------------------------------------

  /**
   * Returns a raw GPU pointer into this PE's local slice.
   * @warning Do NOT dereference on the host without cudaMemcpy.
   */
  const void* access(const MemoryRegion& mrb, Offset off) const override {
    const auto& mr    = static_cast<const MemoryRegionNVSHMEM&>(mrb);
    const int   my_pe = pg_.rank().value();
    return static_cast<const uint8_t*>(mr.symm_base_)
           + mr.pe_byte_offsets_[my_pe]
           + off.value() * mr.elsize_;
  }

  // -- get ------------------------------------------------------------------

  /**
   * Blocking get via GPU-aware MPI_Get.
   *
   * The MPI library reads directly from the remote GPU buffer into @p to_buf
   * (also expected to be a GPU pointer) using RDMA — no host staging.
   *
   * @pre to_buf must be a valid GPU device pointer.
   */
  void get(MemoryRegion& mrb, Proc proc, Offset off, Size nelements, void* to_buf) override {
    auto&        mr        = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target    = proc.value();
    const size_t byte_off  = off.value() * mr.elsize_;
    const size_t nbytes    = nelements.value() * mr.elsize_;
    const MPI_Aint disp    = static_cast<MPI_Aint>(mr.pe_byte_offsets_[target] + byte_off);

    MPI_Get(to_buf, static_cast<int>(nbytes), MPI_BYTE,
            target, disp,
            static_cast<int>(nbytes), MPI_BYTE,
            mr.mpi_win_);
    MPI_Win_flush(target, mr.mpi_win_);
  }

  /**
   * Non-blocking get via GPU-aware MPI_Get.
   * Caller must call fence() to ensure completion.
   */
  void nb_get(MemoryRegion& mrb, Proc proc, Offset off, Size nelements, void* to_buf,
              DataCommunicationHandlePtr data_comm_handle) override {
    auto&        mr       = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target   = proc.value();
    const size_t byte_off = off.value() * mr.elsize_;
    const size_t nbytes   = nelements.value() * mr.elsize_;
    const MPI_Aint disp   = static_cast<MPI_Aint>(mr.pe_byte_offsets_[target] + byte_off);

    data_comm_handle->resetCompletionStatus();
    MPI_Get(to_buf, static_cast<int>(nbytes), MPI_BYTE,
            target, disp,
            static_cast<int>(nbytes), MPI_BYTE,
            mr.mpi_win_);
    // Completion deferred: caller calls fence() -> MPI_Win_flush_all()
  }

  // -- put ------------------------------------------------------------------

  /**
   * Blocking put via GPU-aware MPI_Put.
   * @pre from_buf must be a valid GPU device pointer.
   */
  void put(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
           const void* from_buf) override {
    auto&        mr       = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target   = proc.value();
    const size_t byte_off = off.value() * mr.elsize_;
    const size_t nbytes   = nelements.value() * mr.elsize_;
    const MPI_Aint disp   = static_cast<MPI_Aint>(mr.pe_byte_offsets_[target] + byte_off);

    MPI_Put(from_buf, static_cast<int>(nbytes), MPI_BYTE,
            target, disp,
            static_cast<int>(nbytes), MPI_BYTE,
            mr.mpi_win_);
    MPI_Win_flush(target, mr.mpi_win_);
  }

  /**
   * Non-blocking put via GPU-aware MPI_Put.
   * Completion deferred to fence().
   */
  void nb_put(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
              const void* from_buf, DataCommunicationHandlePtr data_comm_handle) override {
    auto&        mr       = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target   = proc.value();
    const size_t byte_off = off.value() * mr.elsize_;
    const size_t nbytes   = nelements.value() * mr.elsize_;
    const MPI_Aint disp   = static_cast<MPI_Aint>(mr.pe_byte_offsets_[target] + byte_off);

    data_comm_handle->resetCompletionStatus();
    MPI_Put(from_buf, static_cast<int>(nbytes), MPI_BYTE,
            target, disp,
            static_cast<int>(nbytes), MPI_BYTE,
            mr.mpi_win_);
  }

  // -- add (accumulate) -----------------------------------------------------

  /**
   * Blocking accumulate:  remote[off .. off+n] += from_buf[0 .. n]
   *
   * For same-rank self-accumulate: uses a CUDA kernel (no MPI round-trip).
   * For all cross-rank targets:    uses MPI_Accumulate(MPI_SUM) over the
   *   CUDA-aware MPI_Win — the NIC reads from_buf directly from GPU memory.
   *
   * @pre from_buf must be a valid GPU device pointer.
   */
  void add(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
           const void* from_buf) override {
    auto&        mr       = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target   = proc.value();
    const int    my_pe    = pg_.rank().value();
    const size_t byte_off = off.value() * mr.elsize_;
    const size_t n        = nelements.value();

    if(target == my_pe) {
      // Self-accumulate: CUDA kernel, avoids MPI self-messaging overhead
      uint8_t* dst_ptr = static_cast<uint8_t*>(mr.symm_base_)
                         + mr.pe_byte_offsets_[my_pe] + byte_off;
      tamm_nvshmem_kernels::launch_self_accumulate(
        dst_ptr, from_buf, mr.eltype_, n, /*stream=*/0);
      cudaStreamSynchronize(0);
    } else {
      // Cross-rank: GPU-aware MPI_Accumulate; NIC reads from_buf from GPU
      MPI_Datatype mpi_t = to_mpi_type(mr.eltype_);
      const MPI_Aint disp = static_cast<MPI_Aint>(mr.pe_byte_offsets_[target] + byte_off);
      MPI_Accumulate(from_buf, static_cast<int>(n), mpi_t,
                     target, disp,
                     static_cast<int>(n), mpi_t,
                     MPI_SUM, mr.mpi_win_);
      MPI_Win_flush(target, mr.mpi_win_);
    }
  }

  /**
   * Non-blocking accumulate.
   * Self-add is completed immediately (CUDA kernel + stream sync);
   * cross-rank MPI_Accumulate completion is deferred to fence().
   */
  void nb_add(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
              const void* from_buf, DataCommunicationHandlePtr data_comm_handle) override {
    auto&        mr       = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target   = proc.value();
    const int    my_pe    = pg_.rank().value();
    const size_t byte_off = off.value() * mr.elsize_;
    const size_t n        = nelements.value();

    data_comm_handle->resetCompletionStatus();

    if(target == my_pe) {
      uint8_t* dst_ptr = static_cast<uint8_t*>(mr.symm_base_)
                         + mr.pe_byte_offsets_[my_pe] + byte_off;
      tamm_nvshmem_kernels::launch_self_accumulate(
        dst_ptr, from_buf, mr.eltype_, n, /*stream=*/0);
      // stream sync deferred — caller must sync before reading
    } else {
      MPI_Datatype mpi_t = to_mpi_type(mr.eltype_);
      const MPI_Aint disp = static_cast<MPI_Aint>(mr.pe_byte_offsets_[target] + byte_off);
      MPI_Accumulate(from_buf, static_cast<int>(n), mpi_t,
                     target, disp,
                     static_cast<int>(n), mpi_t,
                     MPI_SUM, mr.mpi_win_);
    }
  }

  // -- print_coll -----------------------------------------------------------

  void print_coll(const MemoryRegion& mrb, std::ostream& os) override {
    const auto&  mr    = static_cast<const MemoryRegionNVSHMEM&>(mrb);
    const int    my_pe = pg_.rank().value();
    const size_t n     = mr.local_nelements().value();

    if(mr.eltype_ != ElementType::double_precision) {
      os << "print_coll: only double_precision supported for debug print\n";
      return;
    }

    std::vector<double> host_buf(n);
    cudaMemcpy(host_buf.data(),
               static_cast<const uint8_t*>(mr.symm_base_) + mr.pe_byte_offsets_[my_pe],
               n * sizeof(double), cudaMemcpyDeviceToHost);

    os << "MemoryManagerNVSHMEM PE " << my_pe << " contents:\n";
    for(size_t i = 0; i < n; ++i) {
      os << "  [" << i << "] = " << host_buf[i] << "\n";
    }
    pg_.barrier();
  }

protected:
  explicit MemoryManagerNVSHMEM(ProcGroup pg)
      : MemoryManager{pg, MemoryManagerKind::nvshmem} {
    EXPECTS(pg.is_valid());
  }

  ~MemoryManagerNVSHMEM() = default;

private:
  /// Map TAMM ElementType to the matching MPI_Datatype for MPI_Accumulate.
  static MPI_Datatype to_mpi_type(ElementType t) {
    switch(t) {
      case ElementType::single_precision: return MPI_FLOAT;
      case ElementType::double_precision: return MPI_DOUBLE;
      case ElementType::single_complex:   return MPI_C_FLOAT_COMPLEX;
      case ElementType::double_complex:   return MPI_C_DOUBLE_COMPLEX;
      default: UNREACHABLE(); return MPI_BYTE;
    }
  }

  friend class ExecutionContext;
}; // class MemoryManagerNVSHMEM

} // namespace tamm

#endif // USE_NVSHMEM
