/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/
#ifndef _CUBLAS_SCOPED_HANDLE_HPP_
#define _CUBLAS_SCOPED_HANDLE_HPP_
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include "cublas_helper.hpp"
#include "cublas_handle.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

/**
* @brief NVIDIA advise for handle creation:
https://devtalk.nvidia.com/default/topic/838794/gpu-accelerated libraries/using-cublas-in-different-cuda-streams/
According to NVIDIA: 
1)	It is required that different handles to be used for different devices:
 http://docs.nvidia.com/cuda/cublas/index.html#cublas-context	
2)	It is recommended (but not required, if care is taken) that different handles be used for different host threads: 
http://docs.nvidia.com/cuda/cublas/index.html#thread-safety2changeme
3)	It is neither required nor recommended that different handles be used for different streams on the same device,
 using the same host thread.
**/

class CublasScopedContextHandler {
    sycl::interop_handle& ih;
    CUdevice nativeDevice;
    cublasHandle_t cublasHandle;
    CUstream get_stream(const sycl::queue& queue);
    sycl::context get_context(const sycl::queue& queue);

public:
    CublasScopedContextHandler(sycl::interop_handle& ih);

    /**
   * @brief get_handle: creates the handle by implicitly impose the advice
   * given by nvidia for creating a cublas_handle. (e.g. one cuStream per device
   * per thread).
   * @param queue sycl queue.
   * @return cublasHandle_t a handle to construct cublas routines
   */
    cublasHandle_t get_handle(const sycl::queue& queue);
    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename T, typename U>
    inline T get_mem(U acc) {
        CUdeviceptr cudaPtr = ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc);
        return reinterpret_cast<T>(cudaPtr);
    }

    void wait_stream(const sycl::queue& queue) {
        cuStreamSynchronize(get_stream(queue));
    }
};

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif //_CUBLAS_SCOPED_HANDLE_HPP_
