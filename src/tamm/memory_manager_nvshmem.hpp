#pragma once

/**
 * @file memory_manager_nvshmem.hpp
 * @brief NVSHMEM + CUDA-aware MPI memory manager for TAMM.
 *
 * Replaces MemoryManagerGA for GPU-resident distributed tensors.
 *
 * Communication strategy:
 *   - On-node  (same physical node): nvshmem get/put/atomic over NVLink
 *   - Off-node (remote node):        CUDA-aware MPI one-sided RMA over IB/UCX
 *
 * All pointers are GPU device pointers. No host staging is performed.
 *
 * Usage:
 *   auto* mm = MemoryManagerNVSHMEM::create_coll(pg);
 *   ...
 *   MemoryManagerNVSHMEM::destroy_coll(mm);
 *
 * Prerequisites:
 *   - nvshmemx_init_attr() called before any TAMM init (in tamm.cpp)
 *   - CUDA-aware MPI (OpenMPI built with --with-cuda or equivalent)
 *   - Compile with -DUSE_NVSHMEM
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
// Internal CUDA kernels (file-scope, not exposed in header API)
// ---------------------------------------------------------------------------
namespace tamm_nvshmem_kernels {

/// Element-wise accumulate:  dst[i] += src[i]  for all element types.
template<typename T>
__global__ void accumulate_kernel(T* __restrict__ dst, const T* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) dst[i] += src[i];
}

/// Specialisations for std::complex are not directly CUDA-natively supported,
/// so we alias them through float2 / double2.
__global__ void accumulate_kernel_cf(float2* __restrict__ dst,
                                      const float2* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) {
    dst[i].x += src[i].x;
    dst[i].y += src[i].y;
  }
}

__global__ void accumulate_kernel_cd(double2* __restrict__ dst,
                                      const double2* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) {
    dst[i].x += src[i].x;
    dst[i].y += src[i].y;
  }
}

inline void launch_accumulate(void* dst, const void* src, ElementType eltype,
                               size_t nelements, cudaStream_t stream) {
  const int threads = 256;
  switch(eltype) {
    case ElementType::single_precision: {
      size_t blocks = (nelements + threads - 1) / threads;
      accumulate_kernel<float><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float*>(dst),
        reinterpret_cast<const float*>(src), nelements);
      break;
    }
    case ElementType::double_precision: {
      size_t blocks = (nelements + threads - 1) / threads;
      accumulate_kernel<double><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<double*>(dst),
        reinterpret_cast<const double*>(src), nelements);
      break;
    }
    case ElementType::single_complex: {
      size_t blocks = (nelements + threads - 1) / threads;
      accumulate_kernel_cf<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(dst),
        reinterpret_cast<const float2*>(src), nelements);
      break;
    }
    case ElementType::double_complex: {
      size_t blocks = (nelements + threads - 1) / threads;
      accumulate_kernel_cd<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<double2*>(dst),
        reinterpret_cast<const double2*>(src), nelements);
      break;
    }
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
 * Each rank allocates its local slice from nvshmem_malloc(). A per-rank
 * offset table is exchanged collectively at alloc time so that every rank
 * can compute the symmetric address of any remote element without extra
 * communication.
 *
 * The same GPU buffer is also registered as an MPI_Win (over CUDA-aware MPI)
 * for off-node one-sided RMA.
 */
class MemoryRegionNVSHMEM: public MemoryRegionImpl<MemoryManagerNVSHMEM> {
public:
  explicit MemoryRegionNVSHMEM(MemoryManagerNVSHMEM& mgr)
      : MemoryRegionImpl<MemoryManagerNVSHMEM>(mgr) {}

private:
  // -- symmetric GPU heap ---------------------------------------------------
  void*       symm_base_{nullptr};  ///< nvshmem_malloc'd base (same logical addr on all PEs)
  size_t      symm_total_bytes_{0}; ///< total bytes in symmetric allocation (== npes * local)

  // -- per-rank metadata (host-side) ----------------------------------------
  ElementType         eltype_{ElementType::invalid};
  size_t              elsize_{0};
  std::vector<size_t> pe_byte_offsets_; ///< pe_byte_offsets_[i] = byte offset of PE i's slice

  // -- off-node MPI window (created over GPU memory) ------------------------
  MPI_Win mpi_win_{MPI_WIN_NULL};

  friend class MemoryManagerNVSHMEM;
}; // class MemoryRegionNVSHMEM

// ---------------------------------------------------------------------------
// MemoryManagerNVSHMEM
// ---------------------------------------------------------------------------

/**
 * @ingroup memory_management
 * @brief Distributed memory manager using NVSHMEM (on-node) + CUDA-aware MPI (off-node).
 *
 * Drop-in replacement for MemoryManagerGA that keeps tensor blocks resident
 * in GPU memory end-to-end. Select at compile time with -DUSE_NVSHMEM and
 * at runtime via ExecutionContext::create_coll(pg, MemoryManagerKind::nvshmem).
 */
class MemoryManagerNVSHMEM: public MemoryManager {
public:
  // -- Factory --------------------------------------------------------------

  static MemoryManagerNVSHMEM* create_coll(ProcGroup pg) {
    return new MemoryManagerNVSHMEM{pg};
  }

  static void destroy_coll(MemoryManagerNVSHMEM* mm) { delete mm; }

  // -- Helpers --------------------------------------------------------------

  static size_t get_element_size(ElementType t) {
    switch(t) {
      case ElementType::single_precision: return sizeof(float);
      case ElementType::double_precision: return sizeof(double);
      case ElementType::single_complex:   return sizeof(std::complex<float>);
      case ElementType::double_complex:   return sizeof(std::complex<double>);
      default: UNREACHABLE(); return 0;
    }
  }

  // -- alloc_coll -----------------------------------------------------------

  /**
   * @copydoc MemoryManager::alloc_coll
   *
   * Each rank may contribute a different number of elements (irregular layout).
   * A collective exchange via nvshmem symmetric buffer fills pe_byte_offsets_.
   */
  MemoryRegion* alloc_coll(ElementType eltype, Size local_nelements) override {
    auto* pmr          = new MemoryRegionNVSHMEM(*this);
    pmr->eltype_       = eltype;
    pmr->elsize_       = get_element_size(eltype);
    pmr->local_nelements_ = local_nelements;

    const int    npes          = pg_.size().value();
    const int    my_pe         = pg_.rank().value();
    const size_t local_bytes   = local_nelements.value() * pmr->elsize_;

    // -- 1. Exchange per-PE sizes using a symmetric scratch buffer ----------
    // Each PE writes its own size into a shared symmetric array at index my_pe.
    size_t* symm_sizes = static_cast<size_t*>(nvshmem_malloc(sizeof(size_t) * npes));
    EXPECTS(symm_sizes != nullptr);
    symm_sizes[my_pe] = local_bytes;
    nvshmem_barrier_all();

    // Gather all sizes via nvshmem_size_g (point-to-point, no MPI needed)
    pmr->pe_byte_offsets_.resize(npes + 1);
    std::vector<size_t> pe_sizes(npes);
    for(int pe = 0; pe < npes; ++pe) {
      pe_sizes[pe] = nvshmem_size_g(&symm_sizes[pe], pe);
    }
    nvshmem_free(symm_sizes);

    // Build exclusive prefix-sum -> byte offsets per PE
    pmr->pe_byte_offsets_[0] = 0;
    for(int pe = 0; pe < npes; ++pe) {
      pmr->pe_byte_offsets_[pe + 1] = pmr->pe_byte_offsets_[pe] + pe_sizes[pe];
    }
    pmr->symm_total_bytes_ = pmr->pe_byte_offsets_[npes];

    // -- 2. Allocate the actual symmetric GPU buffer (collective) -----------
    pmr->symm_base_ = nvshmem_malloc(pmr->symm_total_bytes_);
    EXPECTS(pmr->symm_base_ != nullptr);

    if(local_bytes > 0) {
      cudaMemset(static_cast<uint8_t*>(pmr->symm_base_) +
                   pmr->pe_byte_offsets_[my_pe],
                 0, local_bytes);
    }

    // -- 3. Create MPI_Win over the GPU buffer for off-node RMA -------------
    MPI_Win_create(
      static_cast<uint8_t*>(pmr->symm_base_) + pmr->pe_byte_offsets_[my_pe],
      local_bytes,
      1,               // displacement unit = 1 byte
      MPI_INFO_NULL,
      pg_.comm(),
      &pmr->mpi_win_);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, pmr->mpi_win_);

    nvshmem_barrier_all();
    pmr->set_status(AllocationStatus::created);
    return pmr;
  }

  /**
   * @copydoc MemoryManager::alloc_coll_balanced
   *
   * Balanced: all ranks allocate max_nelements. Delegates to alloc_coll.
   */
  MemoryRegion* alloc_coll_balanced(ElementType eltype, Size max_nelements,
                                    ProcList proc_list = {}) override {
    EXPECTS(proc_list.empty());
    return alloc_coll(eltype, max_nelements);
  }

  // -- attach_coll ----------------------------------------------------------

  /**
   * @copydoc MemoryManager::attach_coll
   *
   * Attach creates a second handle to an existing region (shared ownership).
   * The underlying GPU allocation and MPI_Win are *not* duplicated.
   */
  MemoryRegion* attach_coll(MemoryRegion& mrb) override {
    auto& src = static_cast<MemoryRegionNVSHMEM&>(mrb);
    auto* pmr = new MemoryRegionNVSHMEM(*this);

    pmr->symm_base_          = src.symm_base_;
    pmr->symm_total_bytes_   = src.symm_total_bytes_;
    pmr->eltype_             = src.eltype_;
    pmr->elsize_             = src.elsize_;
    pmr->pe_byte_offsets_    = src.pe_byte_offsets_;
    pmr->local_nelements_    = src.local_nelements_;
    pmr->mpi_win_            = src.mpi_win_; // shared handle -- not owned

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
    auto& mr = static_cast<MemoryRegionNVSHMEM&>(mrb);
    mr.symm_base_ = nullptr; // clear reference, do not free
    mr.mpi_win_   = MPI_WIN_NULL;
    pg_.barrier();
  }

  // -- fence ----------------------------------------------------------------

  /**
   * @copydoc MemoryManager::fence
   *
   * Issues nvshmem_quiet (drains all pending NVSHMEM ops) and
   * MPI_Win_flush_all (drains pending MPI RMA ops).
   */
  void fence(MemoryRegion& mrb) override {
    auto& mr = static_cast<MemoryRegionNVSHMEM&>(mrb);
    nvshmem_quiet();
    if(mr.mpi_win_ != MPI_WIN_NULL) {
      MPI_Win_flush_all(mr.mpi_win_);
    }
  }

  // -- access (local pointer) -----------------------------------------------

  /**
   * @copydoc MemoryManager::access
   *
   * Returns a raw GPU pointer into this PE's local slice. Caller must not
   * dereference on the host without explicit cudaMemcpy.
   */
  const void* access(const MemoryRegion& mrb, Offset off) const override {
    const auto& mr       = static_cast<const MemoryRegionNVSHMEM&>(mrb);
    const int   my_pe    = pg_.rank().value();
    return static_cast<const uint8_t*>(mr.symm_base_)
           + mr.pe_byte_offsets_[my_pe]
           + off.value() * mr.elsize_;
  }

  // -- get ------------------------------------------------------------------

  /**
   * @copydoc MemoryManager::get
   *
   * Blocking get. Uses nvshmem on-node (NVLink) or CUDA-aware MPI_Get off-node.
   * @pre to_buf is a valid GPU device pointer.
   */
  void get(MemoryRegion& mrb, Proc proc, Offset off, Size nelements, void* to_buf) override {
    auto&        mr         = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target_pe  = proc.value();
    const size_t byte_off   = off.value() * mr.elsize_;
    const size_t nbytes     = nelements.value() * mr.elsize_;
    uint8_t*     remote_ptr = static_cast<uint8_t*>(mr.symm_base_)
                              + mr.pe_byte_offsets_[target_pe] + byte_off;

    if(is_on_node(target_pe)) {
      nvshmem_getmem(to_buf, remote_ptr, nbytes, target_pe);
    } else {
      MPI_Get(to_buf, nbytes, MPI_BYTE,
              target_pe,
              static_cast<MPI_Aint>(mr.pe_byte_offsets_[target_pe] + byte_off),
              nbytes, MPI_BYTE, mr.mpi_win_);
      MPI_Win_flush(target_pe, mr.mpi_win_);
    }
  }

  /**
   * @copydoc MemoryManager::nb_get
   *
   * Non-blocking get. NVSHMEM path issues a nbi get; MPI path posts MPI_Get
   * (flush deferred to fence()). Callers must call fence() to ensure completion.
   */
  void nb_get(MemoryRegion& mrb, Proc proc, Offset off, Size nelements, void* to_buf,
              DataCommunicationHandlePtr data_comm_handle) override {
    auto&        mr         = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target_pe  = proc.value();
    const size_t byte_off   = off.value() * mr.elsize_;
    const size_t nbytes     = nelements.value() * mr.elsize_;
    uint8_t*     remote_ptr = static_cast<uint8_t*>(mr.symm_base_)
                              + mr.pe_byte_offsets_[target_pe] + byte_off;

    data_comm_handle->resetCompletionStatus();
    if(is_on_node(target_pe)) {
      nvshmem_getmem_nbi(to_buf, remote_ptr, nbytes, target_pe);
    } else {
      MPI_Get(to_buf, nbytes, MPI_BYTE,
              target_pe,
              static_cast<MPI_Aint>(mr.pe_byte_offsets_[target_pe] + byte_off),
              nbytes, MPI_BYTE, mr.mpi_win_);
    }
  }

  // -- put ------------------------------------------------------------------

  /**
   * @copydoc MemoryManager::put
   *
   * Blocking put.
   * @pre from_buf is a valid GPU device pointer.
   */
  void put(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
           const void* from_buf) override {
    auto&        mr         = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target_pe  = proc.value();
    const size_t byte_off   = off.value() * mr.elsize_;
    const size_t nbytes     = nelements.value() * mr.elsize_;
    uint8_t*     remote_ptr = static_cast<uint8_t*>(mr.symm_base_)
                              + mr.pe_byte_offsets_[target_pe] + byte_off;

    if(is_on_node(target_pe)) {
      nvshmem_putmem(remote_ptr, from_buf, nbytes, target_pe);
      nvshmem_quiet();
    } else {
      MPI_Put(from_buf, nbytes, MPI_BYTE,
              target_pe,
              static_cast<MPI_Aint>(mr.pe_byte_offsets_[target_pe] + byte_off),
              nbytes, MPI_BYTE, mr.mpi_win_);
      MPI_Win_flush(target_pe, mr.mpi_win_);
    }
  }

  /**
   * @copydoc MemoryManager::nb_put
   *
   * Non-blocking put. Completion deferred to fence().
   */
  void nb_put(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
              const void* from_buf, DataCommunicationHandlePtr data_comm_handle) override {
    auto&        mr         = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target_pe  = proc.value();
    const size_t byte_off   = off.value() * mr.elsize_;
    const size_t nbytes     = nelements.value() * mr.elsize_;
    uint8_t*     remote_ptr = static_cast<uint8_t*>(mr.symm_base_)
                              + mr.pe_byte_offsets_[target_pe] + byte_off;

    data_comm_handle->resetCompletionStatus();
    if(is_on_node(target_pe)) {
      nvshmem_putmem_nbi(remote_ptr, from_buf, nbytes, target_pe);
    } else {
      MPI_Put(from_buf, nbytes, MPI_BYTE,
              target_pe,
              static_cast<MPI_Aint>(mr.pe_byte_offsets_[target_pe] + byte_off),
              nbytes, MPI_BYTE, mr.mpi_win_);
    }
  }

  // -- add (accumulate) -----------------------------------------------------

  /**
   * @copydoc MemoryManager::add
   *
   * Blocking accumulate:  remote[off..off+n] += from_buf[0..n]
   *
   * On-node:   CUDA kernel adds src into remote GPU buffer directly,
   *            exploiting NVLink peer access (nvshmem symmetric memory is
   *            peer-registered at init, no cudaMemcpyPeer staging needed).
   * Off-node:  MPI_Accumulate(MPI_SUM) over CUDA-aware MPI.
   *
   * @pre from_buf is a valid GPU device pointer.
   */
  void add(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
           const void* from_buf) override {
    auto&        mr        = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target_pe = proc.value();
    const size_t byte_off  = off.value() * mr.elsize_;
    const size_t n         = nelements.value();
    uint8_t*     dst_ptr   = static_cast<uint8_t*>(mr.symm_base_)
                             + mr.pe_byte_offsets_[target_pe] + byte_off;

    if(is_on_node(target_pe)) {
      tamm_nvshmem_kernels::launch_accumulate(
        dst_ptr, from_buf, mr.eltype_, n, /*stream=*/0);
      cudaStreamSynchronize(0);
    } else {
      MPI_Datatype mpi_type = to_mpi_type(mr.eltype_);
      MPI_Accumulate(from_buf, static_cast<int>(n), mpi_type,
                     target_pe,
                     static_cast<MPI_Aint>(mr.pe_byte_offsets_[target_pe] + byte_off),
                     static_cast<int>(n), mpi_type,
                     MPI_SUM, mr.mpi_win_);
      MPI_Win_flush(target_pe, mr.mpi_win_);
    }
  }

  /**
   * @copydoc MemoryManager::nb_add
   *
   * Non-blocking accumulate. Completion deferred to fence().
   */
  void nb_add(MemoryRegion& mrb, Proc proc, Offset off, Size nelements,
              const void* from_buf, DataCommunicationHandlePtr data_comm_handle) override {
    auto&        mr        = static_cast<MemoryRegionNVSHMEM&>(mrb);
    const int    target_pe = proc.value();
    const size_t byte_off  = off.value() * mr.elsize_;
    const size_t n         = nelements.value();
    uint8_t*     dst_ptr   = static_cast<uint8_t*>(mr.symm_base_)
                             + mr.pe_byte_offsets_[target_pe] + byte_off;

    data_comm_handle->resetCompletionStatus();
    if(is_on_node(target_pe)) {
      tamm_nvshmem_kernels::launch_accumulate(
        dst_ptr, from_buf, mr.eltype_, n, /*stream=*/0);
    } else {
      MPI_Datatype mpi_type = to_mpi_type(mr.eltype_);
      MPI_Accumulate(from_buf, static_cast<int>(n), mpi_type,
                     target_pe,
                     static_cast<MPI_Aint>(mr.pe_byte_offsets_[target_pe] + byte_off),
                     static_cast<int>(n), mpi_type,
                     MPI_SUM, mr.mpi_win_);
    }
  }

  // -- print_coll -----------------------------------------------------------

  void print_coll(const MemoryRegion& mrb, std::ostream& os) override {
    const auto& mr       = static_cast<const MemoryRegionNVSHMEM&>(mrb);
    const int   my_pe    = pg_.rank().value();
    const size_t n       = mr.local_nelements().value();

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
  // -- Locality helper ------------------------------------------------------

  /**
   * Returns true if @p pe is on the same physical node as this rank.
   * Uses nvshmem's built-in node team query -- O(1), no MPI call.
   */
  bool is_on_node(int pe) const {
    return nvshmem_team_pe(NVSHMEMX_TEAM_NODE, pe) >= 0;
  }

  // -- MPI type helper ------------------------------------------------------

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
