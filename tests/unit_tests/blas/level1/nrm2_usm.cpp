/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "cblas.h"
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp, typename fp_res>
int test(const device& dev, int N, int incx) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during NRM2:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, dev);
    vector<fp, decltype(ua)> x(ua);
    fp_res result_ref = fp_res(-1);

    rand_vector(x, N, incx);

    // Call Reference NRM2.
    using fp_ref    = typename ref_type_info<fp>::type;
    const int N_ref = N, incx_ref = std::abs(incx);

    result_ref = ::nrm2<fp_ref, fp_res>(&N_ref, (fp_ref*)x.data(), &incx_ref);

    // Call DPC++ NRM2.

    auto result_p = (fp_res*)oneapi::mkl::malloc_shared(64, sizeof(fp_res), dev, cxt);

    try {
#ifdef CALL_RT_API
        done = oneapi::mkl::blas::nrm2(main_queue, N, x.data(), incx, result_p, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, oneapi::mkl::blas::nrm2,
                    (main_queue, N, x.data(), incx, result_p, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during NRM2:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of NRM2:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal(*result_p, result_ref, N, std::cout);

    oneapi::mkl::free_shared(result_p, cxt);
    return (int)good;
}

class Nrm2UsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(Nrm2UsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP((test<float, float>(GetParam(), 1357, 2)));
    EXPECT_TRUEORSKIP((test<float, float>(GetParam(), 1357, 1)));
    EXPECT_TRUEORSKIP((test<float, float>(GetParam(), 1357, -3)));
}
TEST_P(Nrm2UsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP((test<double, double>(GetParam(), 1357, 2)));
    EXPECT_TRUEORSKIP((test<double, double>(GetParam(), 1357, 1)));
    EXPECT_TRUEORSKIP((test<double, double>(GetParam(), 1357, -3)));
}
TEST_P(Nrm2UsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), 1357, 2)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), 1357, 1)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), 1357, -3)));
}
TEST_P(Nrm2UsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), 1357, 2)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), 1357, 1)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), 1357, -3)));
}

INSTANTIATE_TEST_SUITE_P(Nrm2UsmTestSuite, Nrm2UsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace