#include "internal/test_utils.h"
#include "kernels/batch_matmul_kernels_gpu.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test BatchMatmul Kernel") {
    positive_int m = 10_p;
    positive_int n = 10_p;
    positive_int k = 10_p;
    positive_int batch = 5_p;
    int a_seq_length_dim = -1;
    int b_seq_length_dim = -1;
    int seq_length = -1;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape_a = TensorShape{
        TensorDims{FFOrdered{batch, k, m}},
        DataType::FLOAT,
    };
    TensorShape input_shape_b = TensorShape{
        TensorDims{FFOrdered{batch, n, k}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{batch, n, m}},
        DataType::FLOAT,
    };

    GenericTensorAccessorW a_accessor =
        create_random_filled_accessor_w(input_shape_a, allocator);
    GenericTensorAccessorW b_accessor =
        create_random_filled_accessor_w(input_shape_b, allocator);
    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("gpu_forward_kernel") {
      Kernels::BatchMatmul::gpu_forward_kernel(managed_stream.raw_stream(),
                                               managed_handle.raw_handle(),
                                               output_accessor.get_float_ptr(),
                                               a_accessor.get_float_ptr(),
                                               b_accessor.get_float_ptr(),
                                               m.int_from_positive_int(),
                                               n.int_from_positive_int(),
                                               k.int_from_positive_int(),
                                               batch.int_from_positive_int(),
                                               a_seq_length_dim,
                                               b_seq_length_dim,
                                               seq_length);
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorW o_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);
      GenericTensorAccessorW a_grad_accessor =
          allocator.allocate_tensor(input_shape_a);
      GenericTensorAccessorW b_grad_accessor =
          allocator.allocate_tensor(input_shape_b);

      Kernels::BatchMatmul::gpu_backward_kernel(managed_stream.raw_stream(),
                                                managed_handle.raw_handle(),
                                                output_accessor.get_float_ptr(),
                                                o_grad_accessor.get_float_ptr(),
                                                a_accessor.get_float_ptr(),
                                                a_grad_accessor.get_float_ptr(),
                                                b_accessor.get_float_ptr(),
                                                b_grad_accessor.get_float_ptr(),
                                                m.int_from_positive_int(),
                                                n.int_from_positive_int(),
                                                k.int_from_positive_int(),
                                                batch.int_from_positive_int());
    }
  }
}
