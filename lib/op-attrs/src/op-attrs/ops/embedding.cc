#include "op-attrs/ops/embedding.h"
#include "op-attrs/ff_ordered/slice.h"
#include "op-attrs/ff_ordered/transform.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

RecordFormatter as_dot(EmbeddingAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  r << kv("num_entries", attrs.num_entries)
    << kv("out_channels", attrs.out_channels) << kv("aggr", attrs.aggr)
    << kv("output_type", attrs.data_type);

  return r;
}

static std::optional<std::string> basic_check(EmbeddingAttrs const &attrs,
                                              TensorShape const &input) {
  if (input.data_type != DataType::INT32 &&
      input.data_type != DataType::INT64) {
    return fmt::format("Embedding expected input tensor to have integer "
                       "datatype, but receieved tensor of datatype {}",
                       input.data_type);
  }

  if (attrs.aggr != AggregateOp::SUM) {
    return fmt::format(fmt::format(
        "Currently unsupported aggregation op for embedding: {}", attrs.aggr));
  }

  return std::nullopt;
}

tl::expected<TensorShape, std::string>
    get_output_shape(EmbeddingAttrs const &attrs, TensorShape const &input) {
  {
    std::optional<std::string> err_msg = basic_check(attrs, input);
    if (err_msg.has_value()) {
      return tl::unexpected(err_msg.value());
    }
  }

  TensorShape output = input;
  dim_at_idx(output.dims, relative_ff_dim_t{-1}) = attrs.out_channels;
  output.data_type = attrs.data_type;
  return output;
}

tl::expected<TensorShape, std::string>
    get_weights_shape(EmbeddingAttrs const &attrs, TensorShape const &input) {
  {
    std::optional<std::string> err_msg = basic_check(attrs, input);
    if (err_msg.has_value()) {
      return tl::unexpected(err_msg.value());
    }
  }

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{
              attrs.num_entries,
              attrs.out_channels,
          },
      },
      attrs.data_type,
  };
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(EmbeddingAttrs const &attrs,
                     ParallelTensorShape const &input) {

  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree =
      SumDegree{shard_dim_at_idx(input, relative_ff_dim_t{-1}).degree};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_p};
  FFOrdered<positive_int> shard_degrees =
      transform(input.dims.shard_dims,
                [](ShardParallelDim const &d) { return d.degree; });
  shard_degrees.at(relative_ff_dim_t{-1}) = get_discard_copy_degree(input);

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_weights_shape(EmbeddingAttrs const &attrs,
                      ParallelTensorShape const &input) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_weights_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{product(transform(
      ff_ordered_shard_dims(input.dims),
      [](ShardParallelDim const &d) -> positive_int { return d.degree; }))};
  positive_int entry_dim_degree = 1_p;
  positive_int out_channel_degree = get_discard_copy_degree(input);
  FFOrdered<positive_int> shard_degrees = FFOrdered{
      entry_dim_degree,
      out_channel_degree,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

std::vector<InitializerAttrs> get_initializers(
    EmbeddingAttrs const &,
    std::optional<InitializerAttrs> const &maybe_initializer_attrs) {
  InitializerAttrs default_initializer_attrs = InitializerAttrs{
      NormInitializerAttrs{
          /*seed=*/0,
          /*mean=*/0.0,
          /*stddev=*/1.0,
      },
  };

  return {maybe_initializer_attrs.value_or(default_initializer_attrs)};
}

} // namespace FlexFlow
