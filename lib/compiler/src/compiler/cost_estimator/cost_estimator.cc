#include "compiler/cost_estimator/cost_estimator.h"

namespace FlexFlow {

CostEstimator::CostEstimator(std::shared_ptr<ICostEstimator> implementation_ptr)
    : implementation_ptr(implementation_ptr) {}

OpCostMetrics CostEstimator::estimate_cost(OpCostEstimateKey const &k) const {
  return this->implementation_ptr->estimate_cost(k);
}

milliseconds_t CostEstimator::estimate_cost(TensorSetMovement const &m) const {
  return this->implementation_ptr->estimate_cost(m);
}

} // namespace FlexFlow
