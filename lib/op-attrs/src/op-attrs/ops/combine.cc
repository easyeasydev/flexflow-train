#include "op-attrs/ops/combine.h"
#include "op-attrs/ff_dim_t.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

RecordFormatter as_dot(CombineAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  r << kv("dim", attrs.combine_dim) << kv("degree", attrs.combine_degree);

  return r;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(CombineAttrs const &attrs,
                     ParallelTensorShape const &input) {
  ShardParallelDim input_dim = ({
    std::optional<ShardParallelDim> result = try_get_shard_dim_at_idx(
        input, relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim));
    if (!result.has_value()) {
      return tl::unexpected(fmt::format(
          "Failed to get shard dim at index {} in parallel tensor shape {}",
          attrs.combine_dim,
          input));
    }

    result.value();
  });

  if (input_dim.degree % attrs.combine_degree != 0) {
    return tl::unexpected(
        fmt::format("Combine received tensor containing parallel dim {} with "
                    "degree {}, which is not divisible by combine degree {}",
                    attrs.combine_dim,
                    input_dim.degree,
                    attrs.combine_degree));
  }

  ParallelTensorShape output = input;
  relative_ff_dim_t combine_dim =
      relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim);
  shard_dim_at_idx(output, combine_dim).degree = positive_int{
      shard_dim_at_idx(output, combine_dim).degree / attrs.combine_degree};

  return output;
}

} // namespace FlexFlow
