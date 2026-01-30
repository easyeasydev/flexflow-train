#include "models/yolov10/yolov10.h"
#include "models/yolov10/yolov10_config.dtg.h"
#include "pcg/computation_graph.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/repeat.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

YOLOv10Config get_default_yolov10_config() {

  constexpr auto get_default_yolov10_layers =
      []() -> std::vector<YOLOv10LayerConfig> {
    std::vector<YOLOv10LayerConfig> layers{};

    // Add all layers of the default model
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::Conv,
        /*module_args=*/{64, 3, 2},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::Conv,
        /*module_args=*/{128, 3, 2},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/3_p,
        /*module_type=*/YOLOv10Module::C2f,
        /*module_args=*/{128, 1},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::Conv,
        /*module_args=*/{256, 3, 2},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/6_p,
        /*module_type=*/YOLOv10Module::C2f,
        /*module_args=*/{256, 1},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::SCDown,
        /*module_args=*/{512, 3, 2},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/6_p,
        /*module_type=*/YOLOv10Module::C2fCIB,
        /*module_args=*/{512, 1},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::SCDown,
        /*module_args=*/{1024, 3, 2},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/3_p,
        /*module_type=*/YOLOv10Module::C2fCIB,
        /*module_args=*/{1024, 1},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::SPPF,
        /*module_args=*/{1024, 5},
    });
    layers.push_back(YOLOv10LayerConfig{
        /*input_tensor_index=*/{-1},
        /*num_module_repeats=*/1_p,
        /*module_type=*/YOLOv10Module::PSA,
        /*module_args=*/{1024},
    });

    return layers;
  };

  return YOLOv10Config{
      /*num_input_channels=*/3_p,
      /*num_classes=*/80_p,
      /*model_scales=*/{1.0, 1.25, 512},
      /*model_layers=*/get_default_yolov10_layers(),
  };
}

// tensor_guid_t create_yolov10_mlp(ComputationGraphBuilder &cgb,
//                                  YOLOv10Config const &config,
//                                  tensor_guid_t const &input,
//                                  std::vector<positive_int> const &mlp_layers)
//                                  {
//   tensor_guid_t t = input;

//   // Refer to
//   //
//   https://github.com/facebookresearch/yolov10/blob/64063a359596c72a29c670b4fcc9450bb342e764/yolov10_s_pytorch.py#L218-L228
//   // for example initializer.
//   for (size_t i = 0; i < mlp_layers.size() - 1; i++) {
//     float std_dev = sqrt(2.0f / (mlp_layers.at(i + 1) + mlp_layers.at(i)));
//     InitializerAttrs projection_initializer =
//         InitializerAttrs{NormInitializerAttrs{
//             /*seed=*/config.seed,
//             /*mean=*/0,
//             /*stddev=*/std_dev,
//         }};

//     std_dev = sqrt(2.0f / mlp_layers.at(i + 1));
//     InitializerAttrs bias_initializer =
//     InitializerAttrs{NormInitializerAttrs{
//         /*seed=*/config.seed,
//         /*mean=*/0,
//         /*stddev=*/std_dev,
//     }};

//     t = cgb.dense(/*input=*/t,
//                   /*outDim=*/mlp_layers.at(i + 1),
//                   /*activation=*/Activation::RELU,
//                   /*use_bias=*/true,
//                   /*data_type=*/DataType::FLOAT,
//                   /*projection_initializer=*/projection_initializer,
//                   /*bias_initializer=*/bias_initializer);
//   }
//   return t;
// }

// tensor_guid_t create_yolov10_sparse_embedding_network(
//     ComputationGraphBuilder &cgb, YOLOv10Config const &config,
//     tensor_guid_t const &input, positive_int input_dim,
//     positive_int output_dim) {
//   float range = sqrt(1.0f / input_dim);
//   InitializerAttrs embed_initializer =
//   InitializerAttrs{UniformInitializerAttrs{
//       /*seed=*/config.seed,
//       /*min_val=*/-range,
//       /*max_val=*/range,
//   }};

//   tensor_guid_t t = cgb.embedding(input,
//                                   /*num_entries=*/input_dim,
//                                   /*outDim=*/output_dim,
//                                   /*aggr=*/AggregateOp::SUM,
//                                   /*dtype=*/DataType::HALF,
//                                   /*kernel_initializer=*/embed_initializer);
//   return cgb.cast(t, DataType::FLOAT);
// }

// tensor_guid_t create_yolov10_interact_features(
//     ComputationGraphBuilder &cgb, YOLOv10Config const &config,
//     tensor_guid_t const &bottom_mlp_output,
//     std::vector<tensor_guid_t> const &emb_outputs) {
//   if (config.arch_interaction_op != YOLOv10ArchInteractionOp::CAT) {
//     throw mk_runtime_error(fmt::format(
//         "Currently only arch_interaction_op=YOLOv10ArchInteractionOp::CAT is
//         " "supported, but found arch_interaction_op={}. If you need support
//         for " "additional " "arch_interaction_op value, please create an
//         issue.", format_as(config.arch_interaction_op)));
//   }

//   return cgb.concat(
//       /*tensors=*/concat_vectors({bottom_mlp_output}, emb_outputs),
//       /*axis=*/relative_ff_dim_t{1});
// }

void create_yolov10_layer(ComputationGraphBuilder &cgb,
                          YOLOv10Config const &model_config,
                          YOLOv10LayerConfig const &layer_config,
                          std::vector<tensor_guid_t> &tensor_cache) {}

ComputationGraph get_yolov10_computation_graph(YOLOv10Config const &config) {

  ComputationGraphBuilder cgb;

  // Create the initial input tensors
  std::vector<tensor_guid_t> tensor_cache{};
  // ...

  for (size_t i = 0; i < config.model_layers.size(); i++) {
    const YOLOv10LayerConfig layer_config = config.model_layers[i];
    create_yolov10_layer(cgb, config, layer_config, tensor_cache);
  }

  return cgb.computation_graph;
}

// ComputationGraph get_computation_graph(YOLOv10Config const &config) {
//   ComputationGraphBuilder cgb;

//   auto create_input_tensor = [&](FFOrdered<positive_int> const &dims,
//                                  DataType const &data_type) ->
//                                  tensor_guid_t
//                                  {
//     TensorShape input_shape = TensorShape{
//         TensorDims{dims},
//         data_type,
//     };
//     return cgb.create_input(input_shape, CreateGrad::YES);
//   };

//   // Create input tensors
//   std::vector<tensor_guid_t> sparse_inputs =
//       repeat(num_elements(config.embedding_size), [&]() {
//         return create_input_tensor(
//             FFOrdered{config.batch_size, config.embedding_bag_size},
//             DataType::INT64);
//       });

//   tensor_guid_t dense_input = create_input_tensor(
//       FFOrdered{config.batch_size, config.dense_arch_layer_sizes.front()},
//       DataType::FLOAT);

//   // Construct the model
//   tensor_guid_t bottom_mlp_output = create_yolov10_mlp(
//       /*cgb=*/cgb,
//       /*config=*/config,
//       /*input=*/dense_input,
//       /*mlp_layers=*/config.dense_arch_layer_sizes);

//   std::vector<tensor_guid_t> emb_outputs =
//       transform(zip(config.embedding_size, sparse_inputs),
//                 [&](std::pair<positive_int, tensor_guid_t> const
//                 &combined_pair)
//                     -> tensor_guid_t {
//                   return create_yolov10_sparse_embedding_network(
//                       /*cgb=*/cgb,
//                       /*config=*/config,
//                       /*input=*/combined_pair.second,
//                       /*input_dim=*/combined_pair.first,
//                       /*output_dim=*/config.embedding_dim);
//                 });

//   tensor_guid_t interacted_features = create_yolov10_interact_features(
//       /*cgb=*/cgb,
//       /*config=*/config,
//       /*bottom_mlp_output=*/bottom_mlp_output,
//       /*emb_outputs=*/emb_outputs);

//   tensor_guid_t output = create_yolov10_mlp(
//       /*cgb=*/cgb,
//       /*config=*/config,
//       /*input=*/interacted_features,
//       /*mlp_layers=*/config.over_arch_layer_sizes);

//   return cgb.computation_graph;
// }

} // namespace FlexFlow
