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

CublasScopedContextHandler::CublasScopedContextHandler(sycl::interop_handle& ih)
        : ih(ih), nativeDevice(ih.get_native_device<sycl::backend::ext_oneapi_cuda>()) {
    cublasStatus_t err;
    CUBLAS_ERROR_FUNC(cublasCreate, err, &cublasHandle);
}

cublasHandle_t CublasScopedContextHandler::get_handle(const sycl::queue& queue) {
    CUstream streamId = get_stream(queue);
    cudaStream_t currentStreamId;
    cublasStatus_t err;
    CUBLAS_ERROR_FUNC(cublasGetStream, err, cublasHandle, &currentStreamId);
    if (currentStreamId != streamId) {
        CUBLAS_ERROR_FUNC(cublasSetStream, err, cublasHandle, streamId);
    }
    return cublasHandle;
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
