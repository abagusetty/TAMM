#pragma once

#include "tamm/errors.hpp"
#include <vector>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas.h>
#elif defined(USE_DPCPP)
#include "sycl_device.hpp"
#endif

namespace tamm {

constexpr unsigned short int max_gpu_streams = 2;

#if defined(USE_HIP)
using gpuStream_t     = hipStream_t;
using gpuEvent_t      = hipEvent_t;
using gpuBlasHandle_t = rocblas_handle;
#elif defined(USE_CUDA)
using gpuStream_t     = cudaStream_t;
using gpuEvent_t      = cudaEvent_t;
using gpuBlasHandle_t = cublasHandle_t;
#elif defined(USE_DPCPP)
using gpuStream_t = sycl::queue;
using gpuEvent_t  = sycl::event;

auto sycl_asynchandler = [](sycl::exception_list exceptions) {
  for(std::exception_ptr const& e: exceptions) {
    try {
      std::rethrow_exception(e);
    } catch(sycl::exception const& ex) {
      std::cout << "Caught asynchronous SYCL exception:" << std::endl
                << ex.what() << ", SYCL code: " << ex.code() << std::endl;
    }
  }
};
#endif

static inline void getDeviceCount(int* id) {
#if defined(USE_CUDA)
  cudaGetDeviceCount(id);
#elif defined(USE_HIP)
  hipGetDeviceCount(id);
#elif defined(USE_DPCPP)
  syclGetDeviceCount(id);
#endif
}

static inline void gpuSetDevice(int active_device) {
#ifdef USE_CUDA
  cudaSetDevice(active_device);
#elif defined(USE_HIP)
  hipSetDevice(active_device);
#elif defined(USE_DPCPP)
  syclSetDevice(active_device);
#endif
}

class GPUStreamPool {
protected:
  bool _initialized{false};
  // Active GPU set by a given MPI-rank from execution context ctor
  int _active_device;
  // total number of GPUs on node
  int _ngpus{0};

  // Map of GPU-IDs and stream
  std::vector<gpuStream_t*> _devID2Stream;

#if defined(USE_CUDA) || defined(USE_HIP)
  // Map of GPU-IDs and blashandle
  std::vector<gpuBlasHandle_t*> _devID2Handle;
#endif

  // counter for getting a round-robin stream used by (T) code
  unsigned int _count{0};

private:
  GPUStreamPool() {
    getDeviceCount(&_ngpus);

    for(int devID = 0; devID < _ngpus; devID++) { // # of GPUs per node
      _active_device = devID;
      gpuSetDevice(devID);

      for(int streamID = 0; streamID < max_gpu_streams; streamID++) { // # of streams per GPU

        gpuStream_t* stream = nullptr;
#if defined(USE_CUDA)
        stream = new cudaStream_t;
        cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking);

        if(streamID == 0) {
          gpuBlasHandle_t* handle = new gpuBlasHandle_t;
          cublasCreate(handle);
          cublasSetStream(*handle, *stream);
          _devID2Handle.push_back(handle);
        }
#elif defined(USE_HIP)
        stream = new hipStream_t;
        hipStreamCreateWithFlags(stream, hipStreamNonBlocking);

        if(streamID == 0) {
          gpuBlasHandle_t* handle = new gpuBlasHandle_t;
          rocblas_create_handle(handle);
          rocblas_set_stream(*handle, *stream);
          _devID2Handle.push_back(handle);
        }
#elif defined(USE_DPCPP)
        stream = new sycl::queue(*sycl_get_context(devID), *sycl_get_device(devID),
                                 sycl_asynchandler,
                                 sycl::property_list{sycl::property::queue::in_order{}});
#endif

        _devID2Stream.push_back(stream);

      } // streamID
    }   // devID

    _initialized = false;
    _count       = 0;
  }

  ~GPUStreamPool() {
    _initialized = false;
    _count       = 0;

    for(int devID = 0; devID < _ngpus; devID++) { // # of GPUs per node
      gpuSetDevice(devID);

      for(int streamID = 0; streamID < max_gpu_streams; streamID++) { // # of streams per GPU
        gpuStream_t* stream = _devID2Stream[devID * max_gpu_streams + streamID];
#if defined(USE_CUDA)
        cudaStreamDestroy(*stream);

        if(streamID == 0) {
          gpuBlasHandle_t* handle = _devID2Handle[devID];
          cublasDestroy(*handle);
          handle = nullptr;
        }
#elif defined(USE_HIP)
        hipStreamDestroy(*stream);

        if(streamID == 0) {
          gpuBlasHandle_t* handle = _devID2Handle[devID];
          rocblas_destroy_handle(*handle);
          handle = nullptr;
        }
#elif defined(USE_DPCPP)
        delete stream;
#endif

        stream = nullptr;
      } // streamID
    }   // devID

    _devID2Stream.clear();
#if defined(USE_CUDA) || defined(USE_HIP)
    _devID2Handle.clear();
#endif
  }

  void check_device() {
    if(!_initialized) {
      EXPECTS_STR(false, "Error: active GPU-device not set! call set_device()!");
    }
  }

public:
  /// sets an active device for getting streams and blas handles
  void set_device(int device) {
    if(!_initialized) {
      _active_device = device;
      gpuSetDevice(_active_device);
      _initialized = true;
    }
  }

  /// Returns a GPU stream in a round-robin fashion
  // Note: This function is used only in (T) method
  gpuStream_t& getRRStream() {
    check_device();
    unsigned short int counter = _count++ % max_gpu_streams;
    return *(_devID2Stream[_active_device * max_gpu_streams + counter]);
  }

  gpuStream_t& getStream() {
    check_device();
    return *(_devID2Stream[_active_device * max_gpu_streams]);
  }

#if !defined(USE_DPCPP)
  /// Returns a GPU BLAS handle that is valid only for the CUDA and HIP builds
  gpuBlasHandle_t& getBlasHandle() {
    check_device();
    return *(_devID2Handle[_active_device]);
  }
#endif

  /// Returns the instance of device manager singleton.
  inline static GPUStreamPool& getInstance() {
    static GPUStreamPool d_m{};
    return d_m;
  }

  GPUStreamPool(const GPUStreamPool&)            = delete;
  GPUStreamPool& operator=(const GPUStreamPool&) = delete;
  GPUStreamPool(GPUStreamPool&&)                 = delete;
  GPUStreamPool& operator=(GPUStreamPool&&)      = delete;
};

} // namespace tamm
