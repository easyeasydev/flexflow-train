/**
 * @file yolov10.h
 *
 * @brief YOLOv10 detection model
 */

#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_YOLOV10_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_YOLOV10_H

#include "models/yolov10/yolov10_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the YOLOv10 model

/**
 * @brief Get the default YOLOv10 config.
 *
 * @details The configs here refer to the example at
 * https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v10/yolov10x.yaml.
 */
YOLOv10Config get_default_yolov10_config();

// tensor_guid_t create_yolov10_mlp(ComputationGraphBuilder &cgb,
//                                  YOLOv10Config const &config,
//                                  tensor_guid_t const &input,
//                                  std::vector<size_t> const &mlp_layers);

// tensor_guid_t create_yolov10_sparse_embedding_network(
//     ComputationGraphBuilder &cgb, YOLOv10Config const &config,
//     tensor_guid_t const &input, int input_dim, int output_dim);

// tensor_guid_t
// create_yolov10_interact_features(ComputationGraphBuilder &cgb,
//                                  YOLOv10Config const &config,
//                                  tensor_guid_t const &bottom_mlp_output,
//                                  std::vector<tensor_guid_t> const
//                                  &emb_outputs);

void create_yolov10_layer(ComputationGraphBuilder &cgb,
                          YOLOv10Config const &model_config,
                          YOLOv10LayerConfig const &layer_config,
                          std::vector<tensor_guid_t> &tensor_cache);

/**
 * @brief Get the YOLOv10 computation graph.
 *
 * @param YOLOv10Config The config of YOLOv10 model.
 * @return ComputationGraph The computation graph of a YOLOv10 model.
 */
ComputationGraph get_yolov10_computation_graph(YOLOv10Config const &config);

} // namespace FlexFlow

#endif
