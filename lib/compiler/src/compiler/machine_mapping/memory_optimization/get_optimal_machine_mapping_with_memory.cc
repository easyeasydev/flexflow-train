#include "compiler/machine_mapping/memory_optimization/get_optimal_machine_mapping_with_memory.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/get_machine_resource_splits.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_cache.h"
#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_result.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.dtg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/contains.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/exception.h"
#include "utils/overload.h"

namespace FlexFlow {

MachineMappingWithMemoryResult get_optimal_machine_mapping_with_memory(
    MachineMappingWithMemoryCache &result_cache,
    MachineMappingWithMemoryContext const &context,
    MachineMappingProblemTree const &problem_tree,
    MachineSpecification const &resources,
    MachineMappingConstraints const &constraints) {

  MachineMappingState state = MachineMappingState{
      problem_tree,
      resources,
      constraints,
  };

  {
    std::optional<MachineMappingWithMemoryResult> cached_result =
        machine_mapping_with_memory_cache_load(result_cache, state);
    if (cached_result) {
      return cached_result.value();
    }
  }

  MachineMappingWithMemoryResult result =
      problem_tree.visit<MachineMappingWithMemoryResult>(overload{
          [&](MMProblemTreeSeriesSplit const &series_split) {
            return get_optimal_machine_mapping_with_memory(
                result_cache,
                context,
                series_split,
                resources,
                constraints,
                /*parallel_split_transformation=*/std::nullopt);
          },
          [&](auto const &decomp_tree_node) {
            return get_optimal_machine_mapping_with_memory(result_cache,
                                                           context,
                                                           decomp_tree_node,
                                                           resources,
                                                           constraints);
          },
      });

  machine_mapping_with_memory_cache_save(result_cache, state, result);
  return result;
}

MachineMappingWithMemoryResult get_optimal_machine_mapping_with_memory(
    MachineMappingWithMemoryCache &result_cache,
    MachineMappingWithMemoryContext const &context,
    MMProblemTreeSeriesSplit const &series_split,
    MachineSpecification const &resources,
    MachineMappingConstraints const &constraints,
    std::optional<ParallelSplitTransformation> const
        &parallel_split_transformation) {

  auto get_boundary_machine_view_assignments =
      [&](std::unordered_set<BinaryTreePath> const &boundary_layers)
      -> std::unordered_set<ParallelLayerGuidObliviousMachineMapping> {
    std::unordered_map<BinaryTreePath, std::unordered_set<MachineView>>
        allowed = generate_map(
            boundary_layers,
            [&](BinaryTreePath const &l) -> std::unordered_set<MachineView> {
              UnmappedRuntimeOnlyOpCostEstimateKey leaf =
                  mm_problem_tree_get_subtree_at_path(
                      MachineMappingProblemTree{series_split}, l)
                      .value()
                      .get<UnmappedRuntimeOnlyOpCostEstimateKey>();
              return context.allowed_machine_views(leaf, resources);
            });
    return transform(
        get_all_assignments(allowed),
        [](std::unordered_map<BinaryTreePath, MachineView> const &m) {
          return ParallelLayerGuidObliviousMachineMapping{m};
        });
  };

  auto eval_pre_boundary_mapping =
      [&](ParallelLayerGuidObliviousMachineMapping const
              &assigned_pre_machine_views) {
        MachineMappingConstraints pre_candidate = with_additional_constraints(
            restrict_to_left_child(constraints), assigned_pre_machine_views);

        MachineMappingWithMemoryResult pre_result =
            get_optimal_machine_mapping_with_memory(
                result_cache,
                context,
                series_split.get_left_child(),
                resources,
                pre_candidate);

        return pre_result;
      };

  auto eval_post_boundary_mapping =
      [&](ParallelLayerGuidObliviousMachineMapping const
              &assigned_post_machine_views) {
        MachineMappingConstraints post_candidate = with_additional_constraints(
            restrict_to_right_child(constraints), assigned_post_machine_views);

        MachineMappingWithMemoryResult post_result =
            get_optimal_machine_mapping_with_memory(
                result_cache,
                context,
                series_split.get_right_child(),
                resources,
                post_candidate);

        return post_result;
      };

  MachineMappingWithMemoryResult result =
      empty_machine_mapping_with_memory_result();
  AbstractedTensorSetMovement tensor_movement =
      series_split.tensor_set_movement;

  for (ParallelLayerGuidObliviousMachineMapping const
           &assigned_pre_machine_views :
       get_boundary_machine_view_assignments(get_src_layers(tensor_movement))) {

    MachineMappingWithMemoryResult pre_result =
        eval_pre_boundary_mapping(assigned_pre_machine_views);

    for (ParallelLayerGuidObliviousMachineMapping const
             &assigned_post_machine_views :
         get_boundary_machine_view_assignments(
             get_dst_layers(tensor_movement))) {

      MachineMappingWithMemoryResult post_result =
          eval_post_boundary_mapping(assigned_post_machine_views);

      TensorSetMovement comm_across_split =
          concretize_abstracted_tensor_set_movement(
              tensor_movement,
              /*pre_mapping=*/assigned_pre_machine_views,
              /*post_mapping=*/assigned_post_machine_views);
      milliseconds_t cost_across_split =
          context.cost_estimator.estimate_cost(comm_across_split);

      result = minimize_runtime(result,
                                series_combine(cost_across_split,
                                               pre_result,
                                               post_result,
                                               parallel_split_transformation));
    }
  }

  return result;
}

MachineMappingWithMemoryResult get_optimal_machine_mapping_with_memory(
    MachineMappingWithMemoryCache &result_cache,
    MachineMappingWithMemoryContext const &context,
    MMProblemTreeParallelSplit const &parallel_split,
    MachineSpecification const &resources,
    MachineMappingConstraints const &constraints) {

  MachineMappingProblemTree lhs = parallel_split.get_left_child();
  MachineMappingProblemTree rhs = parallel_split.get_right_child();

  MachineMappingWithMemoryResult series_result = [&] {
    MMProblemTreeSeriesSplit series_split = MMProblemTreeSeriesSplit{
        /*tensor_set_movement=*/empty_abstracted_tensor_set_movement(),
        /*left_child=*/lhs,
        /*right_child=*/rhs,
    };

    return get_optimal_machine_mapping_with_memory(
        result_cache,
        context,
        series_split,
        resources,
        constraints,
        ParallelSplitTransformation::LthenR);
  }();

  MachineMappingConstraints left_constraints =
      restrict_to_left_child(constraints);
  MachineMappingConstraints right_constraints =
      restrict_to_right_child(constraints);

  auto evaluate_resource_split =
      [&](std::pair<MachineSpecification, MachineSpecification> const
              &resource_split) {
        MachineMappingWithMemoryResult left_result =
            get_optimal_machine_mapping_with_memory(result_cache,
                                                    context,
                                                    lhs,
                                                    resource_split.first,
                                                    left_constraints);
        MachineMappingWithMemoryResult right_result =
            get_optimal_machine_mapping_with_memory(result_cache,
                                                    context,
                                                    rhs,
                                                    resource_split.second,
                                                    right_constraints);

        return parallel_combine(left_result, right_result);
      };

  std::unordered_set<MachineMappingWithMemoryResult> parallel_results =
      transform(get_machine_resource_splits(resources),
                evaluate_resource_split);

  return minimize_runtime(series_result,
                          get_mapping_with_minimal_runtime(parallel_results));
}

MachineMappingWithMemoryResult get_optimal_machine_mapping_with_memory(
    MachineMappingWithMemoryCache &result_cache,
    MachineMappingWithMemoryContext const &context,
    UnmappedRuntimeOnlyOpCostEstimateKey const &leaf,
    MachineSpecification const &resource,
    MachineMappingConstraints const &constraints) {

  std::unordered_set<MachineView> candidates = [&] {
    std::optional<MachineView> machine_view = require_only_root(constraints);
    if (machine_view.has_value()) {
      return std::unordered_set{machine_view.value()};
    } else {
      return context.allowed_machine_views(leaf, resource);
    }
  }();

  auto get_mapping_result = [&](MachineView const &machine_view) {
    OpCostEstimateKey mapped = map_unmapped_op_cost_estimate_key(
        unmapped_op_cost_estimate_key_from_runtime_only(
            leaf, context.optimizer_attrs),
        machine_view);
    OpCostMetrics cost = context.cost_estimator.estimate_cost(mapped);

    return make_singleton_machine_mapping_with_memory_result(cost,
                                                             machine_view);
  };

  std::unordered_set<MachineMappingWithMemoryResult> candidate_results =
      transform(candidates, get_mapping_result);

  return get_mapping_with_minimal_runtime(candidate_results);
}

} // namespace FlexFlow
