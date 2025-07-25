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
#include "kernels/embedding_kernels_gpu.h"

namespace FlexFlow::Kernels::Embedding {

template <typename TD>
__global__ void rand_generate_int(TD *ptr, size_t size, TD p) {
  CUDA_KERNEL_LOOP(i, size) {
    ptr[i] = i % p;
  }
}

void rand_generate_int64_wrapper(int64_t *ptr, size_t size, int64_t p) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // Randomly initialize the intput tensor to avoid out of index range issues
  rand_generate_int<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      ptr, size, p);
}

void rand_generate_int32_wrapper(int32_t *ptr, size_t size, int32_t p) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // Randomly initialize the intput tensor to avoid out of index range issues
  rand_generate_int<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      ptr, size, p);
}

template <typename TD>
__global__ void embed_forward_no_aggr(int32_t const *input,
                                      TD *output,
                                      TD const *embed,
                                      int out_dim,
                                      int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    int32_t wordIdx = input[idx];
    output[i] = embed[wordIdx * out_dim + off];
  }
}

template <typename TD>
__global__ void embed_forward_no_aggr(int64_t const *input,
                                      TD *output,
                                      TD const *embed,
                                      int out_dim,
                                      int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    int64_t wordIdx = input[idx];
    output[i] = embed[wordIdx * out_dim + off];
  }
}

template <typename TD>
__global__ void embed_forward_with_aggr(int32_t const *input,
                                        TD *output,
                                        TD const *embed,
                                        int out_dim,
                                        int in_dim,
                                        int batch_size,
                                        AggregateOp aggr) {
  TD scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      int32_t wordIdx = input[idx * in_dim + j];
      output[i] = output[i] + embed[wordIdx * out_dim + off];
      if (aggr == AggregateOp::SUM) {
      } else {
        assert(aggr == AggregateOp::AVG);
        output[i] = output[i] * scale;
      }
    }
  }
}

template <typename TD>
__global__ void embed_forward_with_aggr(int64_t const *input,
                                        TD *output,
                                        TD const *embed,
                                        int out_dim,
                                        int in_dim,
                                        int batch_size,
                                        AggregateOp aggr) {
  TD scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
      output[i] = output[i] + embed[wordIdx * out_dim + off];
      if (aggr == AggregateOp::SUM) {
      } else {
        assert(aggr == AggregateOp::AVG);
        output[i] = output[i] * scale;
      }
    }
  }
}

template <typename TD>
__global__ void embed_backward_no_aggr(int32_t const *input,
                                       TD const *output,
                                       TD *embed,
                                       int out_dim,
                                       int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    int32_t wordIdx = input[idx];
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
  }
}

template <typename TD>
__global__ void embed_backward_no_aggr(int64_t const *input,
                                       TD const *output,
                                       TD *embed,
                                       int out_dim,
                                       int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    int64_t wordIdx = input[idx];
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
  }
}

// Specialization for half type

template <>
__global__ void embed_backward_no_aggr<half>(int32_t const *input,
                                             half const *output,
                                             half *embed,
                                             int out_dim,
                                             int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    int32_t wordIdx = input[idx];
#if __CUDA_ARCH__ >= 700
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
#else
    assert(false);
    // TODO: this implementation may result in race condition
    // so we use an assertion failure to warn users
    embed[wordIdx * out_dim + off] += output[i];
#endif
  }
}

template <>
__global__ void embed_backward_no_aggr<half>(int64_t const *input,
                                             half const *output,
                                             half *embed,
                                             int out_dim,
                                             int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    int64_t wordIdx = input[idx];
#if __CUDA_ARCH__ >= 700
    atomicAdd(embed + wordIdx * out_dim + off, output[i]);
#else
    assert(false);
    // TODO: this implementation may result in race condition
    // so we use an assertion failure to warn users
    embed[wordIdx * out_dim + off] += output[i];
#endif
  }
}

template <typename TD>
__global__ void embed_backward_with_aggr(int32_t const *input,
                                         TD const *output,
                                         TD *embed,
                                         int out_dim,
                                         int in_dim,
                                         int batch_size,
                                         AggregateOp aggr) {
  TD scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    TD gradient;
    if (aggr == AggregateOp::SUM) {
      gradient = output[i];
    } else {
      assert(aggr == AggregateOp::AVG);
      gradient = output[i] * scale;
    }
    for (int j = 0; j < in_dim; j++) {
      int32_t wordIdx = input[idx * in_dim + j];
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
    }
  }
}

template <typename TD>
__global__ void embed_backward_with_aggr(int64_t const *input,
                                         TD const *output,
                                         TD *embed,
                                         int out_dim,
                                         int in_dim,
                                         int batch_size,
                                         AggregateOp aggr) {
  TD scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    TD gradient;
    if (aggr == AggregateOp::SUM) {
      gradient = output[i];
    } else {
      assert(aggr == AggregateOp::AVG);
      gradient = output[i] * scale;
    }
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
    }
  }
}

// Specialization for half type

template <>
__global__ void embed_backward_with_aggr<half>(int32_t const *input,
                                               half const *output,
                                               half *embed,
                                               int out_dim,
                                               int in_dim,
                                               int batch_size,
                                               AggregateOp aggr) {
  half scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    half gradient;
    if (aggr == AggregateOp::SUM) {
      gradient = output[i];
    } else {
      assert(aggr == AggregateOp::AVG);
      gradient = output[i] * scale;
    }
    for (int j = 0; j < in_dim; j++) {
      int32_t wordIdx = input[idx * in_dim + j];
#if __CUDA_ARCH__ >= 700
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
#else
      assert(false);
      // TODO: this implementation may result in race condition
      // so we use an assertion failure to warn users
      embed[wordIdx * out_dim + off] += gradient;
#endif
    }
  }
}

template <>
__global__ void embed_backward_with_aggr<half>(int64_t const *input,
                                               half const *output,
                                               half *embed,
                                               int out_dim,
                                               int in_dim,
                                               int batch_size,
                                               AggregateOp aggr) {
  half scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    half gradient;
    if (aggr == AggregateOp::SUM) {
      gradient = output[i];
    } else {
      assert(aggr == AggregateOp::AVG);
      gradient = output[i] * scale;
    }
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
#if __CUDA_ARCH__ >= 700
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
#else
      assert(false);
      // TODO: this implementation may result in race condition
      // so we use an assertion failure to warn users
      embed[wordIdx * out_dim + off] += gradient;
#endif
    }
  }
}

template <DataType TI, DataType TD>
struct ForwardKernel {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  GenericTensorAccessorR const &weight,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    throw mk_runtime_error(fmt::format(
        "Invalid type combination: input type {} and output type {}", TI, TD));
  }
};

template <>
struct ForwardKernel<DataType::INT32, DataType::FLOAT> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  GenericTensorAccessorR const &weight,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_forward_no_aggr<float>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::FLOAT>(),
                       weight.get<DataType::FLOAT>(),
                       out_dim,
                       batch_size);
    } else {
      assert(aggr == AggregateOp::AVG || aggr == AggregateOp::SUM);
      embed_forward_with_aggr<float>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::FLOAT>(),
                       weight.get<DataType::FLOAT>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct ForwardKernel<DataType::INT32, DataType::HALF> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  GenericTensorAccessorR const &weight,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_forward_no_aggr<half>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::HALF>(),
                       weight.get<DataType::HALF>(),
                       out_dim,
                       batch_size);
    } else {
      assert(aggr == AggregateOp::AVG || aggr == AggregateOp::SUM);
      embed_forward_with_aggr<half>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::HALF>(),
                       weight.get<DataType::HALF>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct ForwardKernel<DataType::INT32, DataType::DOUBLE> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  GenericTensorAccessorR const &weight,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_forward_no_aggr<double>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::DOUBLE>(),
                       weight.get<DataType::DOUBLE>(),
                       out_dim,
                       batch_size);
    } else {
      assert(aggr == AggregateOp::AVG || aggr == AggregateOp::SUM);
      embed_forward_with_aggr<double>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::DOUBLE>(),
                       weight.get<DataType::DOUBLE>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct ForwardKernel<DataType::INT64, DataType::FLOAT> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  GenericTensorAccessorR const &weight,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_forward_no_aggr<float>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::FLOAT>(),
                       weight.get<DataType::FLOAT>(),
                       out_dim,
                       batch_size);
    } else {
      assert(aggr == AggregateOp::AVG || aggr == AggregateOp::SUM);
      embed_forward_with_aggr<float>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::FLOAT>(),
                       weight.get<DataType::FLOAT>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct ForwardKernel<DataType::INT64, DataType::HALF> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  GenericTensorAccessorR const &weight,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_forward_no_aggr<half>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::HALF>(),
                       weight.get<DataType::HALF>(),
                       out_dim,
                       batch_size);
    } else {
      assert(aggr == AggregateOp::AVG || aggr == AggregateOp::SUM);
      embed_forward_with_aggr<half>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::HALF>(),
                       weight.get<DataType::HALF>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct ForwardKernel<DataType::INT64, DataType::DOUBLE> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  GenericTensorAccessorR const &weight,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_forward_no_aggr<double>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::DOUBLE>(),
                       weight.get<DataType::DOUBLE>(),
                       out_dim,
                       batch_size);
    } else {
      assert(aggr == AggregateOp::AVG || aggr == AggregateOp::SUM);
      embed_forward_with_aggr<double>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::DOUBLE>(),
                       weight.get<DataType::DOUBLE>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <DataType TI, DataType TD>
struct BackwardKernel {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &weight_grad,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    throw mk_runtime_error(fmt::format(
        "Invalid type combination: input type {} and output type {}", TI, TD));
  }
};

template <>
struct BackwardKernel<DataType::INT32, DataType::FLOAT> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &weight_grad,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_backward_no_aggr<float>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::FLOAT>(),
                       weight_grad.get<DataType::FLOAT>(),
                       out_dim,
                       batch_size);
    } else {
      embed_backward_with_aggr<float>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::FLOAT>(),
                       weight_grad.get<DataType::FLOAT>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct BackwardKernel<DataType::INT32, DataType::DOUBLE> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &weight_grad,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_backward_no_aggr<double>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::DOUBLE>(),
                       weight_grad.get<DataType::DOUBLE>(),
                       out_dim,
                       batch_size);
    } else {
      embed_backward_with_aggr<double>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::DOUBLE>(),
                       weight_grad.get<DataType::DOUBLE>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct BackwardKernel<DataType::INT32, DataType::HALF> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &weight_grad,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_backward_no_aggr<half>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::HALF>(),
                       weight_grad.get<DataType::HALF>(),
                       out_dim,
                       batch_size);
    } else {
      embed_backward_with_aggr<half>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT32>(),
                       output.get<DataType::HALF>(),
                       weight_grad.get<DataType::HALF>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct BackwardKernel<DataType::INT64, DataType::FLOAT> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &weight_grad,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_backward_no_aggr<float>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::FLOAT>(),
                       weight_grad.get<DataType::FLOAT>(),
                       out_dim,
                       batch_size);
    } else {
      embed_backward_with_aggr<float>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::FLOAT>(),
                       weight_grad.get<DataType::FLOAT>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct BackwardKernel<DataType::INT64, DataType::DOUBLE> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &weight_grad,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_backward_no_aggr<double>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::DOUBLE>(),
                       weight_grad.get<DataType::DOUBLE>(),
                       out_dim,
                       batch_size);
    } else {
      embed_backward_with_aggr<double>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::DOUBLE>(),
                       weight_grad.get<DataType::DOUBLE>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

template <>
struct BackwardKernel<DataType::INT64, DataType::HALF> {
  void operator()(cudaStream_t stream,
                  std::optional<AggregateOp> aggr,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &weight_grad,
                  int in_dim,
                  int out_dim,
                  int batch_size) {
    if (!aggr.has_value()) {
      embed_backward_no_aggr<half>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::HALF>(),
                       weight_grad.get<DataType::HALF>(),
                       out_dim,
                       batch_size);
    } else {
      embed_backward_with_aggr<half>
          <<<GET_BLOCKS(
                 get_num_elements(output.shape.dims).int_from_positive_int()),
             CUDA_NUM_THREADS,
             0,
             stream>>>(input.get<DataType::INT64>(),
                       output.get<DataType::HALF>(),
                       weight_grad.get<DataType::HALF>(),
                       out_dim,
                       in_dim,
                       batch_size,
                       aggr.value());
    }
  }
};

void gpu_forward_kernel(ffStream_t stream,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        GenericTensorAccessorR const &weight,
                        DataType input_data_type,
                        DataType output_data_type,
                        std::optional<AggregateOp> aggr,
                        int in_dim,
                        int out_dim,
                        int batch_size) {
  DataTypeDispatch2<ForwardKernel>{}(input_data_type,
                                     output_data_type,
                                     stream,
                                     aggr,
                                     input,
                                     output,
                                     weight,
                                     in_dim,
                                     out_dim,
                                     batch_size);
}

void gpu_backward_kernel(cudaStream_t stream,
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &weight_grad,
                         DataType output_data_type,
                         DataType input_data_type,
                         std::optional<AggregateOp> aggr,
                         int in_dim,
                         int out_dim,
                         int batch_size) {
  DataTypeDispatch2<BackwardKernel>{}(output_data_type,
                                      input_data_type,
                                      stream,
                                      aggr,
                                      output,
                                      input,
                                      weight_grad,
                                      in_dim,
                                      out_dim,
                                      batch_size);
}

} // namespace FlexFlow::Kernels::Embedding
