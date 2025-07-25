/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "internal/device.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/device.h"
#include "kernels/gather_kernels_gpu.h"
#include "op-attrs/ff_dim_t.h"

namespace FlexFlow::Kernels::Gather {

template <typename IndexType>
__global__ void gather_forward(float const *input,
                               IndexType const *index,
                               float *output,
                               coord_t output_size,
                               coord_t stride,
                               coord_t input_dim_size,
                               coord_t output_dim_size) {
  CUDA_KERNEL_LOOP(o, output_size) {
    // output tensor shape: [*, output_dim_size, stride]
    // output tensor stride: [output_dim_size * stride, stride, 1]
    // output tensor index: [outer_index, index_2, left_over]
    // input tensor shape: [*, input_dim_size, stride]
    // input tensor stride: [input_dim_size * stride, stride, 1]
    // the index of the corresponding input tensor should be:
    // [outer_index, index[0], left_over]
    // Therefore, input_index = outer_index * (stride * input_dim_size)
    //                        + index[0] * stride + left_over;
    coord_t outer_index = o / (stride * output_dim_size);
    // coord_t index_2 = (o / stride) % dim_size
    coord_t left_over = o % stride;
    coord_t input_idx =
        outer_index * (stride * input_dim_size) + index[o] * stride + left_over;
    output[o] = input[input_idx];
  }
}

template <typename IndexType>
__global__ void gather_backward(float const *output_grad,
                                IndexType const *index,
                                float *input_grad,
                                coord_t output_size,
                                coord_t stride,
                                coord_t input_dim_size,
                                coord_t output_dim_size) {
  CUDA_KERNEL_LOOP(o, output_size) {
    // output tensor shape: [*, output_dim_size, stride]
    // output tensor stride: [output_dim_size * stride, stride, 1]
    // output tensor index: [outer_index, index_2, left_over]
    // input tensor shape: [*, input_dim_size, stride]
    // input tensor stride: [input_dim_size * stride, stride, 1]
    // the index of the corresponding input tensor should be:
    // [outer_index, index[0], left_over]
    // Therefore, input_index = outer_index * (stride * input_dim_size)
    //                        + index[0] * stride + left_over;
    coord_t outer_index = o / (stride * output_dim_size);
    // coord_t index_2 = (o / stride) % dim_size
    coord_t left_over = o % stride;
    coord_t input_idx =
        outer_index * (stride * input_dim_size) + index[o] * stride + left_over;

    atomicAdd(&input_grad[input_idx], output_grad[o]);
  }
}

template <DataType IndexType>
struct ForwardKernel {
  void operator()(ffStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorR const &index,
                  GenericTensorAccessorW const &output,
                  coord_t output_size,
                  coord_t stride,
                  coord_t input_dim_size,
                  coord_t output_dim_size) {
    gather_forward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
        input.get_float_ptr(),
        index.get<IndexType>(),
        output.get_float_ptr(),
        output_size,
        stride,
        input_dim_size,
        output_dim_size);
  }
};

template <DataType IndexType>
struct BackwardKernel {
  void operator()(ffStream_t stream,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorR const &index,
                  GenericTensorAccessorW const &input_grad,
                  coord_t output_size,
                  coord_t stride,
                  coord_t input_dim_size,
                  coord_t output_dim_size) {
    gather_backward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
        output_grad.get_float_ptr(),
        index.get<IndexType>(),
        input_grad.get_float_ptr(),
        output_size,
        stride,
        input_dim_size,
        output_dim_size);
  }
};

GatherPerDeviceState gpu_init_kernel(PerDeviceFFHandle const &handle,
                                     ff_dim_t dim) {
  return GatherPerDeviceState{
      /*handle=*/handle,
      /*dim=*/dim,
  };
}

void gpu_forward_kernel(ffStream_t stream,
                        GatherPerDeviceState const &m,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorR const &index,
                        GenericTensorAccessorW const &output) {
  checkCUDA(get_legion_stream(&stream));

  std::optional<coord_t> stride = std::nullopt;
  if (m.dim.value == 0_n) {
    stride = 1;
  } else {
    stride = get_num_elements(slice_tensor_dims(output.shape.dims,
                                                add_to_ff_dim(m.dim, -1),
                                                std::nullopt))
                 .int_from_positive_int();
  }

  coord_t output_dim_size =
      dim_at_idx(output.shape.dims, m.dim).int_from_positive_int();
  coord_t input_dim_size =
      dim_at_idx(input.shape.dims, m.dim).int_from_positive_int();

  assert(index.shape.data_type == DataType::INT32 ||
         index.shape.data_type == DataType::INT64);

  DataTypeDispatch1<ForwardKernel>{}(
      index.shape.data_type,
      stream,
      input,
      index,
      output,
      get_num_elements(output.shape.dims).int_from_positive_int(),
      stride.value(),
      input_dim_size,
      output_dim_size);
}

void gpu_backward_kernel(ffStream_t stream,
                         GatherPerDeviceState const &m,
                         GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorR const &index,
                         GenericTensorAccessorW const &input_grad) {
  checkCUDA(get_legion_stream(&stream));

  std::optional<coord_t> stride = std::nullopt;
  if (m.dim.value == 0_n) {
    stride = 1;
  } else {
    stride = get_num_elements(slice_tensor_dims(output_grad.shape.dims,
                                                add_to_ff_dim(m.dim, -1),
                                                std::nullopt))
                 .int_from_positive_int();
  }

  coord_t output_dim_size =
      dim_at_idx(output_grad.shape.dims, m.dim).int_from_positive_int();
  coord_t input_dim_size =
      dim_at_idx(input_grad.shape.dims, m.dim).int_from_positive_int();

  assert(index.shape.data_type == DataType::INT32 ||
         index.shape.data_type == DataType::INT64);

  DataTypeDispatch1<BackwardKernel>{}(
      index.shape.data_type,
      stream,
      output_grad,
      index,
      input_grad,
      get_num_elements(output_grad.shape.dims).int_from_positive_int(),
      stride.value(),
      input_dim_size,
      output_dim_size);
}

} // namespace FlexFlow::Kernels::Gather
