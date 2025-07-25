#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/computation_graph/computation_graph_edge.dtg.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/tensor_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"

namespace FlexFlow {

ComputationGraph make_empty_computation_graph();

std::unordered_set<layer_guid_t> get_layers(ComputationGraph const &);

LayerAddedResult add_layer(
    ComputationGraph &computation_graph,
    LayerAttrs const &attrs,
    std::vector<tensor_guid_t> const &inputs,
    std::vector<tensor_guid_t> const &weights,
    std::optional<std::vector<CreateGrad>> const &outputs = std::nullopt);

LayerAddedResult add_input_layer(ComputationGraph &computation_graph,
                                 TensorShape const &tensor_shape);
LayerAddedResult add_input_layer_with_grad(ComputationGraph &computation_graph,
                                           TensorShape const &tensor_shape);

TensorAttrs get_tensor_attrs(ComputationGraph const &, tensor_guid_t const &);
bool are_tensor_guid_shapes_equivalent(ComputationGraph const &cg,
                                       tensor_guid_t const &t1,
                                       tensor_guid_t const &t2);

std::vector<layer_guid_t> topological_ordering(ComputationGraph const &cg);

std::vector<tensor_guid_t> get_outgoing_tensors(ComputationGraph const &cg,
                                                layer_guid_t n);

std::vector<tensor_guid_t> get_incoming_tensors(ComputationGraph const &cg,
                                                layer_guid_t n);

std::vector<tensor_guid_t> get_incoming_inputs(ComputationGraph const &,
                                               layer_guid_t const &);

std::vector<TensorShape> get_incoming_input_shapes(ComputationGraph const &,
                                                   layer_guid_t const &);

std::vector<tensor_guid_t> get_incoming_weights(ComputationGraph const &,
                                                layer_guid_t const &);

std::unordered_set<tensor_guid_t> get_all_tensors(ComputationGraph const &);
std::unordered_map<tensor_guid_t, TensorAttrs>
    get_all_tensor_attrs(ComputationGraph const &);

std::unordered_set<ComputationGraphEdge>
    get_subgraph_incoming_edges(ComputationGraph const &,
                                std::unordered_set<layer_guid_t> const &);
std::unordered_set<ComputationGraphEdge>
    get_subgraph_outgoing_edges(ComputationGraph const &,
                                std::unordered_set<layer_guid_t> const &);
std::unordered_set<layer_guid_t>
    get_subgraph_successors(ComputationGraph const &,
                            std::unordered_set<layer_guid_t> const &);

LayerAttrs get_layer_attrs(ComputationGraph const &cg, layer_guid_t const &n);

std::unordered_map<layer_guid_t, LayerAttrs>
    get_layer_attrs_mapping(ComputationGraph const &cg);

layer_guid_t get_layer_by_name(ComputationGraph const &cg,
                               std::string const &name);

ComputationGraph without_layer_names(ComputationGraph const &);

bool computation_graphs_are_isomorphic(ComputationGraph const &,
                                       ComputationGraph const &);

std::string as_dot(ComputationGraph const &);
void debug_print_dot(ComputationGraph const &);

} // namespace FlexFlow

#endif
