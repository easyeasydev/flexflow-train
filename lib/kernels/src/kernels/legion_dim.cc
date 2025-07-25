#include "kernels/legion_dim.h"
#include "op-attrs/tensor_dims.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

positive_int dim_at_idx(TensorDims const &tensor_dims,
                        legion_dim_t legion_dim) {
  return dim_at_idx(
      tensor_dims,
      ff_dim_from_legion_dim(legion_dim, get_num_dims(tensor_dims)));
}

positive_int &dim_at_idx(TensorDims &tensor_dims, legion_dim_t legion_dim) {
  return dim_at_idx(
      tensor_dims,
      ff_dim_from_legion_dim(legion_dim, get_num_dims(tensor_dims)));
}

using T = value_type<0>;
template std::set<legion_dim_t> key_range(LegionOrdered<T> const &);

legion_dim_t add_to_legion_dim(legion_dim_t legion_dim, int value) {
  return legion_dim_t{
      nonnegative_int{legion_dim.value.unwrap_nonnegative() + value}};
}

legion_dim_t legion_dim_from_ff_dim(ff_dim_t ff_dim,
                                    nonnegative_int num_dimensions) {
  return legion_dim_t{nonnegative_int{num_dimensions.unwrap_nonnegative() -
                                      ff_dim.value.unwrap_nonnegative() - 1}};
  ;
}

ff_dim_t ff_dim_from_legion_dim(legion_dim_t legion_dim,
                                nonnegative_int num_dimensions) {
  return ff_dim_t{nonnegative_int{num_dimensions.unwrap_nonnegative() -
                                  legion_dim.value.unwrap_nonnegative() - 1}};
}

} // namespace FlexFlow
