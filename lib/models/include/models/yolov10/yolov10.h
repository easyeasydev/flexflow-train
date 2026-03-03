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

struct YOLOv10LayerChannelTensor {
  positive_int channels_;
  tensor_guid_t tensor_;
};

bool is_yolov10_repeat_module(YOLOv10Module module_type);

tensor_guid_t create_yolov10_tensor(ComputationGraphBuilder &cgb,
                                    FFOrdered<positive_int> const &dims,
                                    DataType const &data_type);

YOLOv10LayerChannelTensor create_yolov10_layer(
    ComputationGraphBuilder &cgb,
    YOLOv10Config const &model_config,
    YOLOv10LayerConfig const &layer_config,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache);

/**
 * @brief Get the YOLOv10 computation graph.
 *
 * @param YOLOv10Config The config of YOLOv10 model.
 * @return ComputationGraph The computation graph of a YOLOv10 model.
 */
ComputationGraph get_yolov10_computation_graph(YOLOv10Config const &config);

} // namespace FlexFlow

#endif
