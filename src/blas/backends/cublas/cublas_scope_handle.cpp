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
#include "cublas_scope_handle.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
thread_local std::shared_ptr<std::unordered_map<CUdevice, cublasHandle_t>>
    CublasScopedContextHandler::device_handle_map{ nullptr };

CublasScopedContextHandler::CublasScopedContextHandler(sycl::queue queue, sycl::interop_handle& ih)
        : ih(ih) {
    if (!device_handle_map) {
        device_handle_map = std::shared_ptr<std::unordered_map<CUdevice, cublasHandle_t>>(
            new std::unordered_map<CUdevice, cublasHandle_t>(), [](auto* map) {
                cublasStatus_t err;
                CUresult cuErr;
                CUcontext primaryCtx;
                for (auto& handle_pair : *map) {
                    CUdevice currentDevice{handle_pair.first};
                    CUDA_ERROR_FUNC(cuDevicePrimaryCtxRetain, cuErr, &primaryCtx, currentDevice);
                    CUDA_ERROR_FUNC(cuCtxSetCurrent, cuErr, primaryCtx);
                    
                    cublasHandle_t& handle = handle_pair.second;
                    CUBLAS_ERROR_FUNC(cublasDestroy, err, handle);
                }
                delete map;
            });
    }
}

cublasHandle_t CublasScopedContextHandler::get_handle(const sycl::queue& queue) {
    CUdevice device = ih.get_native_device<sycl::backend::ext_oneapi_cuda>();
    CUstream streamId = get_stream(queue);
    cublasStatus_t err;

    auto it = device_handle_map->find(device);
    if (it != device_handle_map->end()) {
        cublasHandle_t handle = it->second;
        cudaStream_t currentStreamId;
        CUBLAS_ERROR_FUNC(cublasGetStream, err, handle, &currentStreamId);
        if (currentStreamId != streamId) {
            CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);
        }
        return handle;
    }

    cublasHandle_t handle;
    CUBLAS_ERROR_FUNC(cublasCreate, err, &handle);
    CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);

    (*device_handle_map)[device] = handle;

    return handle;
}

CUstream CublasScopedContextHandler::get_stream(const sycl::queue& queue) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
}
sycl::context CublasScopedContextHandler::get_context(const sycl::queue& queue) {
    return queue.get_context();
}

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
