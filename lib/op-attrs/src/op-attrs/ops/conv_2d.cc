#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/ops/conv_2d/conv_2d_input_shape.h"
#include "op-attrs/ops/conv_2d/conv_2d_parallel_input_shape.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_conv2d_incoming_tensor_roles(Conv2DAttrs const &attrs) {
  std::vector<IncomingTensorRole> result = {
      IncomingTensorRole::INPUT,
      IncomingTensorRole::WEIGHT,
  };

  if (attrs.use_bias) {
    result.push_back(IncomingTensorRole::WEIGHT);
  }

  return result;
}

TensorShape get_kernel_shape(Conv2DAttrs const &attrs,
                             TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          attrs.out_channels,
          input.num_channels,
          attrs.kernel_h,
          attrs.kernel_w,
      }},
      input.datatype,
  };
}

TensorShape get_bias_shape(Conv2DAttrs const &attrs,
                           TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels},
      },
      input.datatype,
  };
}

static positive_int calculate_output_size(positive_int input_size,
                                          nonnegative_int padding_size,
                                          positive_int kernel_size,
                                          positive_int stride) {
  int input_size_raw = input_size.int_from_positive_int();
  int padding_raw = padding_size.unwrap_nonnegative();
  int kernel_size_raw = kernel_size.int_from_positive_int();
  int stride_raw = stride.int_from_positive_int();

  return positive_int{
      (input_size_raw + (2 * padding_raw) - kernel_size_raw) / stride_raw + 1};
}

TensorShape get_output_shape(Conv2DAttrs const &attrs,
                             TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  positive_int out_height =
      calculate_output_size(/*input_size=*/input.height,
                            /*padding_size=*/attrs.padding_h,
                            /*kernel_size=*/attrs.kernel_h,
                            /*stride_size=*/attrs.stride_h);
  positive_int out_width =
      calculate_output_size(/*input_size=*/input.width,
                            /*padding_size=*/attrs.padding_w,
                            /*kernel_size=*/attrs.kernel_w,
                            /*stride_size=*/attrs.stride_w);

  return TensorShape{TensorDims{FFOrdered<positive_int>{
                         input.num_samples,
                         attrs.out_channels,
                         out_height,
                         out_width,
                     }},
                     input.datatype};
}

std::vector<TensorShape> get_weight_shapes(Conv2DAttrs const &attrs,
                                           TensorShape const &input_shape) {
  std::vector<TensorShape> weight_shapes = {
      get_kernel_shape(attrs, input_shape),
  };

  if (attrs.use_bias) {
    weight_shapes.push_back(get_bias_shape(attrs, input_shape));
  }

  return weight_shapes;
}

ParallelTensorShape get_kernel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_kernel_shape(attrs, get_reduced_shape(input));

  assert(parsed.height_dim.degree == 1);
  assert(parsed.width_dim.degree == 1);

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{parsed.sample_dim.degree * parsed.sum_reduction_degree};
  FFOrdered<positive_int> shard_degrees = FFOrdered{
      parsed.discard_copy_reduction_degree,
      parsed.channel_dim.degree,
      1_p,
      1_p,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

ParallelTensorShape get_bias_shape(Conv2DAttrs const &attrs,
                                   ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_bias_shape(attrs, get_reduced_shape(input));

  SumDegree sum_degree =
      SumDegree{parsed.sum_reduction_degree * parsed.channel_dim.degree};
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{parsed.height_dim.degree * parsed.width_dim.degree *
                        parsed.sample_dim.degree};
  FFOrdered<positive_int> shard_degrees = FFOrdered{
      parsed.discard_copy_reduction_degree,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

ParallelTensorShape get_output_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_output_shape(attrs, get_reduced_shape(input));

  assert(parsed.height_dim.degree == 1);
  assert(parsed.width_dim.degree == 1);

  SumDegree sum_degree =
      SumDegree{parsed.sum_reduction_degree * parsed.channel_dim.degree};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_p};
  FFOrdered<positive_int> shard_degrees = FFOrdered{
      parsed.sample_dim.degree,
      parsed.discard_copy_reduction_degree,
      1_p,
      1_p,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

std::vector<ParallelTensorShape>
    get_weight_shapes(Conv2DAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {
  std::vector<ParallelTensorShape> weight_shapes = {
      get_kernel_shape(attrs, input_shape),
  };

  if (attrs.use_bias) {
    weight_shapes.push_back(get_bias_shape(attrs, input_shape));
  }

  return weight_shapes;
}

/**
 * @brief Chosen to match pytorch implementation
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/conv.py#L178-L187
 */
std::vector<InitializerAttrs>
    get_initializers(Conv2DAttrs const &attrs,
                     TensorShape const &input_shape,
                     std::optional<InitializerAttrs> maybe_kernel_initializer,
                     std::optional<InitializerAttrs> maybe_bias_initializer) {

  if (!attrs.use_bias && maybe_bias_initializer.has_value()) {
    throw mk_runtime_error(fmt::format(
        "Unexpectedly received bias initializer while use_bias=false: {}",
        maybe_bias_initializer));
  }

  TensorShape kernel_shape = get_kernel_shape(attrs, input_shape);

  InitializerAttrs kernel_default_initializer =
      InitializerAttrs{KaimingNormalAttrs{
          /*a=*/sqrtf(5.0),
          /*mode=*/KaimingInitializerMode::FAN_IN,
          /*nonlinearity=*/KaimingInitializerNonlinearity::LEAKY_RELU,
          /*seed=*/0,
      }};

  InitializerAttrs kernel_initializer =
      maybe_kernel_initializer.value_or(kernel_default_initializer);

  positive_int fan_in =
      calculate_fan_for_mode(kernel_shape.dims, KaimingInitializerMode::FAN_IN);

  float bound = 1 / sqrtf(static_cast<float>(fan_in.int_from_positive_int()));

  InitializerAttrs bias_default_initializer =
      InitializerAttrs{UniformInitializerAttrs{
          /*seed=*/0,
          /*min_val=*/-bound,
          /*max_val=*/bound,
      }};

  InitializerAttrs bias_initializer =
      maybe_bias_initializer.value_or(bias_default_initializer);

  if (attrs.use_bias) {
    return {kernel_initializer, bias_initializer};
  } else {
    return {kernel_initializer};
  }
}

} // namespace FlexFlow
