#include "internal/test_utils.h"
#include "kernels/softmax_kernels_gpu.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Softmax Kernel Operations") {
    nonnegative_int input_n = 1_n;
    nonnegative_int input_c = 1_n;
    nonnegative_int input_h = 1_n;
    nonnegative_int input_w = 100_n;
    nonnegative_int channels = 100_n;

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{100_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    SoftmaxPerDeviceState state =
        Kernels::Softmax::gpu_init_kernel(managed_handle.raw_handle(),
                                          ff_dim_t{3_n},
                                          input_n.unwrap_nonnegative(),
                                          channels.unwrap_nonnegative(),
                                          input_h.unwrap_nonnegative(),
                                          input_w.unwrap_nonnegative());

    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorW input_accessor =
          create_random_filled_accessor_w(input_shape, allocator);

      Kernels::Softmax::gpu_forward_kernel(managed_stream.raw_stream(),
                                           state,
                                           input_accessor.get_float_ptr(),
                                           output_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Softmax::gpu_backward_kernel(
          managed_stream.raw_stream(),
          output_grad_accessor.get_float_ptr(),
          input_grad_accessor.get_float_ptr(),
          get_num_elements(output_grad_accessor.shape.dims)
              .int_from_positive_int());

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
