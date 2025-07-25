#ifndef _FLEXFLOW_TEST_COST_ESTIMATOR_FOR_TEST_H
#define _FLEXFLOW_TEST_COST_ESTIMATOR_FOR_TEST_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "compiler/cost_estimator/op_cost_estimate_key.dtg.h"
#include "compiler/cost_estimator/tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.dtg.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.dtg.h"
#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

struct TestCostEstimator : public ICostEstimator {
  std::function<OpCostMetrics(OpCostEstimateKey const &)> get_operator_cost;
  std::function<milliseconds_t(TensorSetMovement const &)>
      get_communication_cost;

  TestCostEstimator() = delete;
  TestCostEstimator(decltype(get_operator_cost) const &get_operator_cost,
                    decltype(get_communication_cost)
                        const &get_communication_cost);

  OpCostMetrics estimate_cost(OpCostEstimateKey const &) const override;

  milliseconds_t estimate_cost(TensorSetMovement const &) const override;
};

CostEstimator make_fake_cost_estimator(
    std::function<OpCostMetrics(OpCostEstimateKey const &)> const
        &get_operator_cost,
    std::function<milliseconds_t(TensorSetMovement const &)> const
        &get_communication_cost);

CostEstimator make_fake_cost_estimator(
    std::unordered_map<OpCostEstimateKey, OpCostMetrics> const &op_cost_map,
    std::unordered_map<TensorSetMovement, milliseconds_t> const &comm_cost_map);

CostEstimator make_fake_constant_cost_estimator(milliseconds_t forward_op_cost,
                                                milliseconds_t backward_op_cost,
                                                milliseconds_t comm_cost,
                                                num_bytes_t memory_cost);

} // namespace FlexFlow

#endif
