#include "models/yolov10/yolov10.h"
#include "models/yolov10/yolov10_config.dtg.h"
#include "models/yolov10/yolov10_module.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/repeat.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/positive_int/positive_int.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace FlexFlow {

namespace {

template <typename T, typename... Ts>
constexpr bool is_one_of(T value, Ts... values) {
  return ((value == values) || ...);
}

positive_int get_module_num_repeats(positive_int num_repeats_in_config,
                                    float model_scales_depth) {
  if (num_repeats_in_config > 1) {
    return positive_int(
        std::max(int(std::round(num_repeats_in_config.int_from_positive_int() *
                                model_scales_depth)),
                 1));
  }

  return num_repeats_in_config;
}

int make_divisible(int input, int divisor) {
  return ((input + divisor - 1) / divisor) * divisor;
}

nonnegative_int autopad_for_yolov10_conv(int kernel_size, int dilation) {
  int const k_eff =
      (dilation > 1) ? (dilation * (kernel_size - 1) + 1) : kernel_size;
  int const p = k_eff / 2;
  return nonnegative_int(p);
}

template <typename T>
T get_arg_or_default(std::vector<int> const &args, size_t idx, T default_val) {
  return (idx < args.size()) ? T(args[idx]) : default_val;
}

} // namespace

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
      /*batch_size=*/64_p,
  };
}

// tensor_guid_t create_yolov10_mlp(ComputationGraphBuilder &cgb,
//                                  YOLOv10Config const &config,
//                                  tensor_guid_t const &input,
//                                  std::vector<positive_int> const &mlp_layers)
//                                  {
//   tensor_guid_t t = input;
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

bool is_yolov10_repeat_module(YOLOv10Module module_type) {
  if (is_one_of(module_type, YOLOv10Module::C2f, YOLOv10Module::C2fCIB)) {
    return true;
  }
  return false;
}

// TODO
YOLOv10LayerChannelTensor create_yolov10_concat_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    std::vector<int> const &input_tensor_index) {
  return {1_p, cgb.identity(layers_cache.front().tensor_)};
}
YOLOv10LayerChannelTensor create_yolov10_detect_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    std::vector<int> const &input_tensor_index) {
  return {1_p, cgb.identity(layers_cache.front().tensor_)};
}
YOLOv10LayerChannelTensor create_yolov10_upsample_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    std::vector<int> const &input_tensor_index) {
  return {1_p, cgb.identity(layers_cache.front().tensor_)};
}

YOLOv10LayerChannelTensor
    create_yolov10_conv_module(ComputationGraphBuilder &cgb,
                               tensor_guid_t const &input_tensor,
                               positive_int const &channel_in,
                               std::vector<int> const &conv_module_args) {

  // Get conv parameters
  // clang-format off
  positive_int channel_out = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/1, /*default_val=*/channel_in);
  positive_int kernel_size = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/2, /*default_val=*/1_p);
  positive_int stride = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/3, /*default_val=*/1_p);
  positive_int groups = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/4, /*default_val=*/1_p);
  bool use_activation = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/5, /*default_val=*/true);
  positive_int dilation = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/6, /*default_val=*/1_p);
  nonnegative_int padding = get_arg_or_default(/*args=*/conv_module_args, /*idx=*/7, /*default_val=*/autopad_for_yolov10_conv(
                                                                                         /*kernel_size=*/kernel_size.int_from_positive_int(),
                                                                                         /*dilation=*/dilation.int_from_positive_int()));
  // clang-format on

  // Create conv layer
  tensor_guid_t conv = cgb.conv2d(
      /*input=*/input_tensor,
      /*outChannels=*/channel_out,
      /*kernelH=*/kernel_size,
      /*kernelW=*/kernel_size,
      /*strideH=*/stride,
      /*strideW=*/stride,
      /*paddingH=*/padding,
      /*paddingW=*/padding,
      /*activation=*/std::nullopt,
      /*groups=*/groups,
      /*use_bias=*/false);

  // Add batch norm and activation
  // TODO: YOLOv10 uses SiLU
  tensor_guid_t out = cgb.batch_norm(
      /*input=*/conv,
      /*affine=*/true,
      /*activation=*/
      use_activation ? std::make_optional(Activation::RELU) : std::nullopt,
      /*eps=*/1e-5,
      /*momentum=*/0.1);

  return {
      .channels_ = channel_out,
      .tensor_ = out,
  };
}

YOLOv10LayerChannelTensor
    create_yolov10_scdown_module(ComputationGraphBuilder &cgb,
                                 tensor_guid_t const &input_tensor,
                                 positive_int const &channel_in,
                                 std::vector<int> const &scdown_module_args) {

  std::vector<int> conv1_module_args = scdown_module_args;
  conv1_module_args[2] = 1; // Change kernel size to 1
  conv1_module_args[3] = 1; // Change stride to 1

  std::vector<int> conv2_module_args = scdown_module_args;
  conv2_module_args.push_back(conv2_module_args[1]); // groups = channel_out
  conv2_module_args.push_back(0);                    // use_activation = false

  YOLOv10LayerChannelTensor conv1 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/input_tensor,
      /*channel_in=*/channel_in,
      /*conv_module_args=*/conv1_module_args);

  YOLOv10LayerChannelTensor conv2 = create_yolov10_conv_module(
      /*cgb=*/cgb,
      /*input_tensor=*/conv1.tensor_,
      /*channel_in=*/conv1.channels_,
      /*conv_module_args=*/conv2_module_args);

  return conv2;
}

YOLOv10LayerChannelTensor create_yolov10_base_module_layer(
    ComputationGraphBuilder &cgb,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache,
    YOLOv10Module module_type,
    std::vector<int> const &input_tensor_index,
    positive_int const &num_module_repeats,
    std::vector<int> const &module_args) {

  if (module_type == YOLOv10Module::Conv) {
    return create_yolov10_conv_module(
        /*cgb=*/cgb,
        /*input_tensor=*/layers_cache.back().tensor_,
        /*channel_in=*/layers_cache.back().channels_,
        /*conv_module_args=*/module_args);
  }

  if (module_type == YOLOv10Module::SCDown) {
    return create_yolov10_scdown_module(
        /*cgb=*/cgb,
        /*input_tensor=*/layers_cache.back().tensor_,
        /*channel_in=*/layers_cache.back().channels_,
        /*conv_module_args=*/module_args);
  }

  if (module_type == YOLOv10Module::C2f) {
    return {1_p, cgb.identity(layers_cache.front().tensor_)};
  }

  return {1_p, cgb.identity(layers_cache.front().tensor_)};
}

// TODO

tensor_guid_t create_yolov10_tensor(ComputationGraphBuilder &cgb,
                                    FFOrdered<positive_int> const &dims,
                                    DataType const &data_type) {
  TensorShape input_shape = TensorShape{
      TensorDims{dims},
      data_type,
  };
  return cgb.create_input(input_shape, CreateGrad::YES);
};

YOLOv10LayerChannelTensor create_yolov10_layer(
    ComputationGraphBuilder &cgb,
    YOLOv10Config const &model_config,
    YOLOv10LayerConfig const &layer_config,
    std::vector<YOLOv10LayerChannelTensor> const &layers_cache) {

  if (layer_config.module_type == YOLOv10Module::Concat) {
    return create_yolov10_concat_layer(
        cgb, layers_cache, layer_config.input_tensor_index);
  }

  if (layer_config.module_type == YOLOv10Module::v10Detect) {
    return create_yolov10_detect_layer(
        cgb, layers_cache, layer_config.input_tensor_index);
  }

  if (layer_config.module_type == YOLOv10Module::Upsample) {
    return create_yolov10_upsample_layer(
        cgb, layers_cache, layer_config.input_tensor_index);
  }

  // Handle other base modules below

  float model_scales_depth = model_config.model_scales.at(0);
  float model_scales_width = model_config.model_scales.at(1);
  int model_scales_max_channels = model_config.model_scales.at(2);

  positive_int num_module_repeats = get_module_num_repeats(
      layer_config.num_module_repeats, model_scales_depth);

  // Get number of input and output channels
  int input_tensor_index = layer_config.input_tensor_index.at(0);
  if (input_tensor_index == -1) {
    input_tensor_index = layers_cache.size() - 1;
  }

  int const channel_in =
      layers_cache.at(input_tensor_index).channels_.int_from_positive_int();

  int channel_out = layer_config.module_args.at(0);
  if (channel_out != model_config.num_classes) {
    // Scale the output channel size if needed
    channel_out = make_divisible(
        std::min(channel_out, model_scales_max_channels) * model_scales_width,
        8);
  }

  // Prepare module args
  std::vector<int> module_args{channel_in, channel_out};
  module_args.insert(module_args.end(),
                     layer_config.module_args.begin() + 1,
                     layer_config.module_args.end());

  if (is_yolov10_repeat_module(layer_config.module_type)) {
    // "Repeat" modules take the number of repeats as one of its arguments
    module_args.insert(module_args.begin() + 2,
                       num_module_repeats.int_from_positive_int());
    num_module_repeats = 1_p;
  }

  return create_yolov10_base_module_layer(
      /*cgb=*/cgb,
      /*layers_cache=*/layers_cache,
      /*module_type=*/layer_config.module_type,
      /*input_tensor_index=*/layer_config.input_tensor_index,
      /*num_module_repeats=*/num_module_repeats,
      /*module_args=*/module_args);
}

ComputationGraph get_yolov10_computation_graph(YOLOv10Config const &config) {

  ComputationGraphBuilder cgb;

  // Create the initial input tensor
  tensor_guid_t input = create_yolov10_tensor(
      cgb,
      FFOrdered{config.batch_size, config.num_input_channels},
      DataType::FLOAT);

  // Cache holding layer-wise information
  std::vector<YOLOv10LayerChannelTensor> layers_cache{YOLOv10LayerChannelTensor{
      .channels_ = config.num_input_channels,
      .tensor_ = input,
  }};

  for (size_t i = 0; i < config.model_layers.size(); i++) {
    const YOLOv10LayerConfig layer_config = config.model_layers[i];
    const YOLOv10LayerChannelTensor layer =
        create_yolov10_layer(cgb, config, layer_config, layers_cache);

    if (i == 0) {
      layers_cache.clear();
    }

    layers_cache.push_back(layer);
  }

  return cgb.computation_graph;
}

//////////////////////////////
//////////////////////////////
//////////////////////////////

// ComputationGraph get_computation_graph(YOLOv10Config const &config) {
//   ComputationGraphBuilder cgb;

//   auto create_input_tensor = [&](FFOrdered<positive_int> const &dims,
//                                  DataType const &data_type) -> tensor_guid_t
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
