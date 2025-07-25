#include "kernels/managed_per_device_ff_handle.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test ManagedPerDeviceFFHandle") {
    ManagedPerDeviceFFHandle base_handle{
        /*num_ranks=*/1,
        /*my_rank=*/0,
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true,
    };
    PerDeviceFFHandle const *base_handle_ptr = &base_handle.raw_handle();

    SUBCASE("constructor") {
      CHECK(base_handle.raw_handle().workSpaceSize == 1024 * 1024);
      CHECK(base_handle.raw_handle().allowTensorOpMathConversion == true);
    }

    SUBCASE("move constructor") {
      ManagedPerDeviceFFHandle new_handle(std::move(base_handle));
      CHECK(&new_handle.raw_handle() == base_handle_ptr);
    }

    SUBCASE("move assignment operator") {
      SUBCASE("move assign to other") {
        ManagedPerDeviceFFHandle new_handle{
            /*num_ranks=*/1,
            /*my_rank=*/0,
            /*workSpaceSize=*/1024 * 1024,
            /*allowTensorOpMathConversion=*/true,
        };
        new_handle = std::move(base_handle);
        CHECK(&new_handle.raw_handle() == base_handle_ptr);
      }

      SUBCASE("move assign to self") {
        base_handle = std::move(base_handle);
        CHECK(&base_handle.raw_handle() == base_handle_ptr);
      }
    }
  }
}
