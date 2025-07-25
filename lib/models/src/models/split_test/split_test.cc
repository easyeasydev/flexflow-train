#include "models/split_test/split_test.h"
#include "pcg/computation_graph_builder.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

ComputationGraph get_split_test_computation_graph(positive_int batch_size) {
  ComputationGraphBuilder cgb;

  positive_int layer_dim1 = 256_p;
  positive_int layer_dim2 = 128_p;
  positive_int layer_dim3 = 64_p;
  positive_int layer_dim4 = 32_p;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<positive_int>{
          batch_size,
          layer_dim1,
      }},
      DataType::FLOAT,
  };

  tensor_guid_t t = cgb.create_input(input_shape, CreateGrad::YES);
  t = cgb.dense(t, layer_dim2);
  t = cgb.relu(t);
  tensor_guid_t t1 = cgb.dense(t, layer_dim3);
  tensor_guid_t t2 = cgb.dense(t, layer_dim3);
  t = cgb.add(t1, t2);
  t = cgb.relu(t);
  t1 = cgb.dense(t, layer_dim4);
  t2 = cgb.dense(t, layer_dim4);
  t = cgb.add(t1, t2);
  t = cgb.relu(t);
  t = cgb.softmax(t);

  return cgb.computation_graph;
}

} // namespace FlexFlow
