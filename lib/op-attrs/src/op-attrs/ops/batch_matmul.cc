#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

// bool BatchMatmulAttrs::is_valid(
//                                 ParallelTensorShape const &lhs,
//                                 ParallelTensorShape const &rhs) const {
//   if (!lhs.is_valid() || !rhs.is_valid()) {
//     return false;
//   }
//   if (lhs.num_dims() != rhs.num_dims()) {
//     return false;
//   }
//   for (int i = lhs.num_dims() - 1; i >= 2; i--) {
//     if (lhs.at(i) != rhs.at(i)) {
//       return false;
//     }
//   }
//   if (lhs.at(0) != rhs.at(1)) {
//     return false;
//   }
//
//   return true;
// }

bool is_valid(BatchMatmulAttrs const &,
              ParallelTensorShape const &,
              ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

tl::expected<TensorShape, std::string>
    get_output_shape(BatchMatmulAttrs const &attrs,
                     TensorShape const &input_lhs,
                     TensorShape const &input_rhs) {
  // If input_lhs is a (b×n×m) tensor,
  // input_rhs is a (b×m×p) tensor,
  // out will be a (b×n×p) tensor.
  // https://pytorch.org/docs/stable/generated/torch.bmm.html

  if (get_num_dims(input_lhs.dims) != 3) {
    return tl::unexpected(
        fmt::format("LHS input has incorrect number of shard dims: {} != {}",
                    get_num_dims(input_lhs.dims),
                    3));
  }
  if (get_num_dims(input_rhs.dims) != 3) {
    return tl::unexpected(
        fmt::format("RHS input has incorrect number of shard dims: {} != {}",
                    get_num_dims(input_rhs.dims),
                    3));
  }
  if (input_lhs.data_type != input_rhs.data_type) {
    return tl::unexpected(fmt::format("Input datatypes do not match: {} != {}",
                                      input_lhs.data_type,
                                      input_rhs.data_type));
  }

  positive_int lhs_b = dim_at_idx(input_lhs.dims, relative_ff_dim_t{0});
  positive_int n = dim_at_idx(input_lhs.dims, relative_ff_dim_t{1});
  positive_int lhs_m = dim_at_idx(input_lhs.dims, relative_ff_dim_t{2});

  positive_int rhs_b = dim_at_idx(input_rhs.dims, relative_ff_dim_t{0});
  positive_int rhs_m = dim_at_idx(input_rhs.dims, relative_ff_dim_t{1});
  positive_int p = dim_at_idx(input_rhs.dims, relative_ff_dim_t{2});

  if (lhs_b != rhs_b) {
    return tl::unexpected(
        fmt::format("LHS b dim ({}) != RHS b dim ({})", lhs_b, rhs_b));
  }
  if (lhs_m != rhs_m) {
    return tl::unexpected(
        fmt::format("RHS m dim ({}) != RHS m dim ({})", lhs_m, rhs_m));
  }

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{
              lhs_b,
              n,
              p,
          },
      },
      input_lhs.data_type,
  };
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(BatchMatmulAttrs const &attrs,
                     ParallelTensorShape const &input_lhs,
                     ParallelTensorShape const &input_rhs) {
  if (num_shard_dims(input_lhs) != 3) {
    return tl::unexpected(
        fmt::format("LHS input has incorrect number of shard dims: {} != {}",
                    num_shard_dims(input_lhs),
                    3));
  }
  if (num_shard_dims(input_rhs) != 3) {
    return tl::unexpected(
        fmt::format("RHS input has incorrect number of shard dims: {} != {}",
                    num_shard_dims(input_rhs),
                    3));
  }
  if (input_lhs.data_type != input_rhs.data_type) {
    return tl::unexpected(fmt::format("Input datatypes do not match: {} != {}",
                                      input_lhs.data_type,
                                      input_rhs.data_type));
  }

  assert(get_total_parallel_degree(input_lhs) ==
         get_total_parallel_degree(input_rhs));

  ShardParallelDim lhs_b = shard_dim_at_idx(input_lhs, relative_ff_dim_t{0});
  ShardParallelDim n = shard_dim_at_idx(input_lhs, relative_ff_dim_t{1});
  ShardParallelDim lhs_m = shard_dim_at_idx(input_lhs, relative_ff_dim_t{2});

  ShardParallelDim rhs_b = shard_dim_at_idx(input_rhs, relative_ff_dim_t{0});
  ShardParallelDim rhs_m = shard_dim_at_idx(input_rhs, relative_ff_dim_t{1});
  ShardParallelDim p = shard_dim_at_idx(input_rhs, relative_ff_dim_t{2});

  if (lhs_b != rhs_b) {
    return tl::unexpected(
        fmt::format("LHS b dim ({}) != RHS b dim ({})", lhs_b, rhs_b));
  }

  if (lhs_m != rhs_m) {
    return tl::unexpected(
        fmt::format("LHS m dim ({}) != RHS m dim ({})", lhs_m, rhs_m));
  }

  if (get_discard_copy_degree(input_lhs) !=
      get_sum_degree(input_rhs) * p.degree) {
    return tl::unexpected(fmt::format("Unexpected number of replicas in LHS: "
                                      "lhs.= ({}) != rhs.+ ({}) * rhs.p ({})",
                                      get_discard_copy_degree(input_lhs),
                                      get_sum_degree(input_rhs),
                                      p.degree));
  }

  if (get_discard_copy_degree(input_rhs) !=
      get_sum_degree(input_lhs) * n.degree) {
    return tl::unexpected(fmt::format("Unexpected number of replicas in RHS: "
                                      "rhs.= ({}) != lhs.+ ({}) * lhs.n ({})",
                                      get_discard_copy_degree(input_rhs),
                                      get_sum_degree(input_lhs),
                                      n.degree));
  }

  ShardParallelDim output_b = lhs_b;
  ShardParallelDim output_n = n;
  ShardParallelDim output_p = p;

  positive_int output_discard_copy_degree = 1_p;
  positive_int output_sum_degree =
      positive_int{get_total_parallel_degree(input_lhs) /
                   (output_b.degree * output_n.degree * output_p.degree)};

  ParallelTensorShape result = ParallelTensorShape{
      ParallelTensorDims{
          FFOrdered<ShardParallelDim>{
              output_b,
              output_n,
              output_p,
          },
          ReplicaParallelDimSet{
              SumDegree{output_sum_degree},
              DiscardCopyDegree{output_discard_copy_degree},
          },
      },
      input_lhs.data_type,
  };

  return result;
}

} // namespace FlexFlow
