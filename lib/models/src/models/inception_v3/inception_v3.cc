#include "models/inception_v3/inception_v3.h"
#include "models/inception_v3/inception_v3_output.dtg.h"
#include "op-attrs/tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

struct CheckShape {
  CheckShape(ComputationGraphBuilder const &cgb,
             InceptionV3Config const &config)
      : cgb(cgb), config(config) {}

  ComputationGraphBuilder const &cgb;
  InceptionV3Config const &config;

  void operator()(tensor_guid_t t,
                  positive_int c,
                  positive_int h,
                  positive_int w) const {
    TensorShape current_shape = cgb.get_shape(t);
    TensorShape expected_shape = TensorShape{
        TensorDims{FFOrdered<positive_int>{
            config.batch_size,
            c,
            h,
            w,
        }},
        DataType::FLOAT,
    };

    if (current_shape != expected_shape) {
      throw mk_runtime_error(fmt::format(
          "Expected activation shape {}, but found activation shape {}",
          expected_shape,
          current_shape));
    }
  }

  void operator()(tensor_guid_t t, positive_int c) const {
    TensorShape current_shape = cgb.get_shape(t);
    TensorShape expected_shape = TensorShape{
        TensorDims{FFOrdered<positive_int>{
            config.batch_size,
            c,
        }},
        DataType::FLOAT,
    };

    if (current_shape != expected_shape) {
      throw mk_runtime_error(fmt::format(
          "Expected activation shape {}, but found activation shape {}",
          expected_shape,
          current_shape));
    }
  }
};

InceptionV3Config get_default_inception_v3_training_config() {
  return InceptionV3Config{
      /*num_classes=*/1000_p,

      // see section 8 of https://arxiv.org/abs/1512.00567 for the source of the
      // batch size
      /*batch_size=*/32_p,

      // see section 4 of https://arxiv.org/abs/1512.00567 for a discussion of
      // auxiliary logits. they are used by default in training
      /*aux_logits=*/true,
  };
}

static tensor_guid_t create_conv_block(ComputationGraphBuilder &cgb,
                                       tensor_guid_t const &input,
                                       positive_int filters,
                                       positive_int kernel_size_h,
                                       positive_int kernel_size_w,
                                       positive_int stride_h = 1_p,
                                       positive_int stride_w = 1_p,
                                       nonnegative_int padding_h = 0_n,
                                       nonnegative_int padding_w = 0_n,
                                       bool use_bias = false) {
  tensor_guid_t conv = cgb.conv2d(input,
                                  /*outChannels=*/filters,
                                  /*kernelH=*/kernel_size_h,
                                  /*kernelW=*/kernel_size_w,
                                  /*strideH=*/stride_h,
                                  /*strideW=*/stride_w,
                                  /*paddingH=*/padding_h,
                                  /*paddingW=*/padding_w,
                                  /*activation=*/std::nullopt,
                                  /*groups=*/1_p,
                                  /*use_bias=*/use_bias);
  return cgb.batch_norm(conv,
                        /*affine=*/true,
                        /*activation=*/Activation::RELU,
                        /*eps=*/1e-5,
                        /*momentum=*/0.1);
}

static tensor_guid_t create_inception_module_a(ComputationGraphBuilder &cgb,
                                               tensor_guid_t const &input,
                                               positive_int pool_features) {
  tensor_guid_t branch1x1 = create_conv_block(cgb,
                                              input,
                                              /*filters=*/64_p,
                                              /*kernel_size_h=*/1_p,
                                              /*kernel_size_w=*/1_p);

  tensor_guid_t branch5x5 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/48_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/64_p,
                          /*kernel_size_h=*/5_p,
                          /*kernel_size_w=*/5_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/2_n,
                          /*padding_w=*/2_n);
    return t;
  }();

  tensor_guid_t branch3x3dbl = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/64_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/96_p,
                          /*kernel_size_h=*/3_p,
                          /*kernel_size_w=*/3_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/1_n,
                          /*padding_w=*/1_n);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/96_p,
                          /*kernel_size_h=*/3_p,
                          /*kernel_size_w=*/3_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/1_n,
                          /*padding_w=*/1_n);
    return t;
  }();

  tensor_guid_t branch_pool = [&] {
    tensor_guid_t t = input;
    t = cgb.pool2d(t,
                   /*kernelH=*/3_p,
                   /*kernelW=*/3_p,
                   /*strideH=*/1_p,
                   /*strideW=*/1_p,
                   /*paddingH=*/1_n,
                   /*paddingW=*/1_n,
                   /*type=*/PoolOp::AVG);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/pool_features,
                          /*kernel_stride_h=*/1_p,
                          /*kernel_stride_w=*/1_p);
    return t;
  }();

  return cgb.concat({branch1x1, branch5x5, branch3x3dbl, branch_pool},
                    /*axis=*/relative_ff_dim_t{1});
}

static tensor_guid_t create_inception_module_b(ComputationGraphBuilder &cgb,
                                               tensor_guid_t const &input) {
  tensor_guid_t branch3x3 = create_conv_block(cgb,
                                              input,
                                              /*filters=*/384_p,
                                              /*kernel_size_h=*/3_p,
                                              /*kernel_size_w=*/3_p,
                                              /*stride_h=*/2_p,
                                              /*stride_w=*/2_p);

  tensor_guid_t branch3x3dbl = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/64_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/96_p,
                          /*kernel_size_h=*/3_p,
                          /*kernel_size_w=*/3_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/1_n,
                          /*padding_w=*/1_n);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/96_p,
                          /*kernel_stride_h=*/3_p,
                          /*kernel_stride_w=*/3_p,
                          /*stride_h=*/2_p,
                          /*stride_w=*/2_p);
    return t;
  }();

  tensor_guid_t branch_pool = cgb.pool2d(input,
                                         /*kernelH=*/3_p,
                                         /*kernelW=*/3_p,
                                         /*strideH=*/2_p,
                                         /*strideW=*/2_p,
                                         /*paddingH=*/0_n,
                                         /*paddingW=*/0_n,
                                         /*type=*/PoolOp::MAX);

  return cgb.concat({branch3x3, branch3x3dbl, branch_pool},
                    /*axis=*/relative_ff_dim_t{1});
}

static tensor_guid_t create_inception_module_c(ComputationGraphBuilder &cgb,
                                               CheckShape const &check_shape,
                                               tensor_guid_t const &input,
                                               positive_int channels_7x7) {
  tensor_guid_t branch1x1 = create_conv_block(cgb,
                                              input,
                                              /*filters=*/192_p,
                                              /*kernel_size_h=*/1_p,
                                              /*kernel_size_w=*/1_p);
  check_shape(branch1x1, 192_p, 17_p, 17_p);

  tensor_guid_t branch7x7 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/channels_7x7,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/channels_7x7,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/7_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/0_n,
                          /*padding_w=*/3_n);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/7_p,
                          /*kernel_size_w=*/1_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/3_n,
                          /*padding_w=*/0_n);
    return t;
  }();
  check_shape(branch7x7, 192_p, 17_p, 17_p);

  tensor_guid_t branch7x7dbl = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/channels_7x7,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/channels_7x7,
                          /*kernel_size_h=*/7_p,
                          /*kernel_size_w=*/1_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/3_n,
                          /*padding_w=*/0_n);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/channels_7x7,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/7_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/0_n,
                          /*padding_w=*/3_n);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/channels_7x7,
                          /*kernel_size_h=*/7_p,
                          /*kernel_size_w=*/1_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/3_n,
                          /*padding_w=*/0_n);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/7_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/0_n,
                          /*padding_w=*/3_n);
    return t;
  }();
  check_shape(branch7x7dbl, 192_p, 17_p, 17_p);

  tensor_guid_t branch_pool = [&] {
    tensor_guid_t t = input;
    t = cgb.pool2d(t,
                   /*kernelH=*/3_p,
                   /*kernelW=*/3_p,
                   /*strideH=*/1_p,
                   /*strideW=*/1_p,
                   /*paddingH=*/1_n,
                   /*paddingW=*/1_n,
                   /*type=*/PoolOp::AVG);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    return t;
  }();
  check_shape(branch_pool, 192_p, 17_p, 17_p);

  return cgb.concat({branch1x1, branch7x7, branch7x7dbl, branch_pool},
                    /*axis=*/relative_ff_dim_t{1});
}

static tensor_guid_t create_inception_module_d(ComputationGraphBuilder &cgb,
                                               tensor_guid_t const &input) {
  tensor_guid_t branch3x3 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    t = create_conv_block(cgb, t, 320_p, 3_p, 3_p, 2_p, 2_p);
    return t;
  }();

  tensor_guid_t branch7x7x3 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/7_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/0_n,
                          /*padding_w=*/3_n);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/7_p,
                          /*kernel_size_w=*/1_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/3_n,
                          /*padding_w=*/0_n);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/3_p,
                          /*kernel_size_w=*/3_p,
                          /*stride_h=*/2_p,
                          /*stride_w=*/2_p);
    return t;
  }();

  tensor_guid_t branch_pool = cgb.pool2d(input,
                                         /*kernelH=*/3_p,
                                         /*kernelW=*/3_p,
                                         /*strideH=*/2_p,
                                         /*strideW=*/2_p,
                                         /*paddingH=*/0_n,
                                         /*paddingW=*/0_n,
                                         /*type=*/PoolOp::MAX);

  return cgb.concat({branch3x3, branch7x7x3, branch_pool},
                    /*axis=*/relative_ff_dim_t{1});
}

static tensor_guid_t create_inception_module_e(ComputationGraphBuilder &cgb,
                                               tensor_guid_t const &input) {
  tensor_guid_t branch1x1 = create_conv_block(cgb,
                                              input,
                                              /*filters=*/320_p,
                                              /*kernel_size_h=*/1_p,
                                              /*kernel_size_w=*/1_p);

  tensor_guid_t branch3x3 = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/384_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    tensor_guid_t t_1 = create_conv_block(cgb,
                                          t,
                                          /*filters=*/384_p,
                                          /*kernel_size_h=*/1_p,
                                          /*kernel_size_w=*/3_p,
                                          /*stride_h=*/1_p,
                                          /*stride_w=*/1_p,
                                          /*padding_h=*/0_n,
                                          /*padding_w=*/1_n);
    tensor_guid_t t_2 = create_conv_block(cgb,
                                          t,
                                          /*filters=*/384_p,
                                          /*kernel_size_h=*/3_p,
                                          /*kernel_size_w=*/1_p,
                                          /*stride_h=*/1_p,
                                          /*stride_w=*/1_p,
                                          /*padding_h=*/1_n,
                                          /*padding_w=*/0_n);
    t = cgb.concat({t_1, t_2}, /*axis=*/relative_ff_dim_t{1});
    return t;
  }();

  tensor_guid_t branch3x3dbl = [&] {
    tensor_guid_t t = input;
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/448_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/384_p,
                          /*kernel_size_h=*/3_p,
                          /*kernel_size_w=*/3_p,
                          /*stride_h=*/1_p,
                          /*stride_w=*/1_p,
                          /*padding_h=*/1_n,
                          /*padding_w=*/1_n);
    tensor_guid_t t_1 = create_conv_block(cgb,
                                          t,
                                          /*filters=*/384_p,
                                          /*kernel_size_h=*/1_p,
                                          /*kernel_size_w=*/3_p,
                                          /*stride_h=*/1_p,
                                          /*stride_w=*/1_p,
                                          /*padding_h=*/0_n,
                                          /*padding_w=*/1_n);
    tensor_guid_t t_2 = create_conv_block(cgb,
                                          t,
                                          /*filters=*/384_p,
                                          /*kernel_size_h=*/3_p,
                                          /*kernel_size_w=*/1_p,
                                          /*stride_h=*/1_p,
                                          /*stride_w=*/1_p,
                                          /*padding_h=*/1_n,
                                          /*padding_w=*/0_n);
    t = cgb.concat({t_1, t_2}, /*axis=*/relative_ff_dim_t{1});
    return t;
  }();

  tensor_guid_t branch_pool = [&] {
    tensor_guid_t t = input;
    t = cgb.pool2d(t,
                   /*kernelH=*/3_p,
                   /*kernelW=*/3_p,
                   /*strideH=*/1_p,
                   /*strideW=*/1_p,
                   /*paddingH=*/1_n,
                   /*paddingW=*/1_n,
                   /*type=*/PoolOp::AVG);
    t = create_conv_block(cgb,
                          t,
                          /*filters=*/192_p,
                          /*kernel_size_h=*/1_p,
                          /*kernel_size_w=*/1_p);
    return t;
  }();

  return cgb.concat({branch1x1, branch3x3, branch3x3dbl, branch_pool},
                    /*axis=*/relative_ff_dim_t{1});
}

static tensor_guid_t create_initial_layers(ComputationGraphBuilder &cgb,
                                           CheckShape const &check_shape,
                                           tensor_guid_t const &input) {
  tensor_guid_t t = input;

  check_shape(t, 3_p, 299_p, 299_p);

  // Conv2d_1a_3x3
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/32_p,
                        /*kernel_size_h=*/3_p,
                        /*kernel_size_w=*/3_p,
                        /*stride_h=*/2_p,
                        /*stride_w=*/2_p);
  check_shape(t, 32_p, 149_p, 149_p);

  // Conv2d_2a_3x3
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/32_p,
                        /*kernel_size_h=*/3_p,
                        /*kernel_size_w=*/3_p);
  check_shape(t, 32_p, 147_p, 147_p);

  // Conv2d_2b_3x3
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/64_p,
                        /*kernel_size_h=*/3_p,
                        /*kernel_size_w=*/3_p,
                        /*stride_h=*/1_p,
                        /*stride_w=*/1_p,
                        /*padding_h=*/1_n,
                        /*padding_w=*/1_n);
  check_shape(t, 64_p, 147_p, 147_p);

  // maxpool1
  t = cgb.pool2d(t,
                 /*kernelH=*/3_p,
                 /*kernelW=*/3_p,
                 /*strideH=*/2_p,
                 /*strideW=*/2_p,
                 /*paddingH=*/0_n,
                 /*paddingW=*/0_n,
                 /*type=*/PoolOp::MAX);
  check_shape(t, 64_p, 73_p, 73_p);

  // Conv2d_3b_1x1
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/80_p,
                        /*kernel_size_h=*/1_p,
                        /*kernel_size_w=*/1_p);
  check_shape(t, 80_p, 73_p, 73_p);

  // Conv2d_4a_3x3
  t = create_conv_block(cgb,
                        t,
                        /*filters=*/192_p,
                        /*kernel_size_h=*/3_p,
                        /*kernel_size_w=*/3_p);
  check_shape(t, 192_p, 71_p, 71_p);

  // maxpool2
  t = cgb.pool2d(t,
                 /*kernelH=*/3_p,
                 /*kernelW=*/3_p,
                 /*strideH=*/2_p,
                 /*strideW=*/2_p,
                 /*paddingH=*/0_n,
                 /*paddingW=*/0_n,
                 /*type=*/PoolOp::MAX);
  check_shape(t, 192_p, 35_p, 35_p);

  return t;
}

static tensor_guid_t create_final_layers(ComputationGraphBuilder &cgb,
                                         CheckShape const &check_shape,
                                         tensor_guid_t const &input,
                                         positive_int num_classes) {
  // avgpool
  tensor_guid_t x = cgb.pool2d(input,
                               /*kernelH=*/8_p,
                               /*kernelW=*/8_p,
                               /*strideH=*/1_p,
                               /*strideW=*/1_p,
                               /*paddingH=*/0_n,
                               /*paddingW=*/0_n,
                               /*type=*/PoolOp::AVG);
  check_shape(x, 2048_p, 1_p, 1_p);

  // dropout
  x = cgb.dropout(x,
                  /*rate=*/0.5);
  check_shape(x, 2048_p, 1_p, 1_p);

  x = cgb.flat(x,
               /*start_dim=*/relative_ff_dim_t{1});
  check_shape(x, 2048_p);

  // fc
  x = cgb.dense(x,
                /*outDim=*/num_classes);
  check_shape(x, num_classes);

  // softmax (not in pytorch model, but shown in Table 1 on p6 of
  // https://arxiv.org/abs/1512.00567)
  x = cgb.softmax(x);
  check_shape(x, num_classes);

  return x;
}

static tensor_guid_t create_inception_aux(ComputationGraphBuilder &cgb,
                                          CheckShape const &check_shape,
                                          tensor_guid_t const &input,
                                          positive_int num_classes) {
  tensor_guid_t x = input;
  check_shape(x, 768_p, 17_p, 17_p);

  x = cgb.pool2d(x,
                 /*kernelH=*/5_p,
                 /*kernelW=*/5_p,
                 /*strideH=*/3_p,
                 /*strideW=*/3_p,
                 /*paddingH=*/0_n,
                 /*paddingW=*/0_n,
                 /*type=*/PoolOp::AVG);
  check_shape(x, 768_p, 5_p, 5_p);

  // conv0
  x = create_conv_block(cgb,
                        x,
                        /*filters=*/128_p,
                        /*kernel_size_h=*/1_p,
                        /*kernel_size_w=*/1_p);
  check_shape(x, 128_p, 5_p, 5_p);

  // conv1
  x = create_conv_block(cgb,
                        x,
                        /*filters=*/768_p,
                        /*kernel_size_h=*/5_p,
                        /*kernel_size_w=*/5_p);
  check_shape(x, 768_p, 1_p, 1_p);

  x = cgb.adaptive_pool2d(x,
                          /*output_h=*/1_p,
                          /*output_w=*/1_p);
  check_shape(x, 768_p, 1_p, 1_p);

  x = cgb.flat(x,
               /*start_dim=*/relative_ff_dim_t{1});
  check_shape(x, 768_p);

  // fc
  x = cgb.dense(x,
                /*outDim=*/num_classes);
  check_shape(x, num_classes);

  return x;
}

static InceptionV3Output create_inception_v3(ComputationGraphBuilder &cgb,
                                             InceptionV3Config const &config,
                                             tensor_guid_t const &input) {
  // NOTE: the shapes for check_shape (as well as the layer names in comments)
  // are pulled from
  // https://github.com/pytorch/vision/blob/6d7851bd5e2bedc294e40e90532f0e375fcfee04/torchvision/models/inception.py#L103-L155
  CheckShape check_shape = CheckShape{
      /*cgb=*/cgb,
      /*config=*/config,
  };

  tensor_guid_t x = create_initial_layers(cgb, check_shape, input);
  check_shape(x, 192_p, 35_p, 35_p);

  // Mixed_5b
  x = create_inception_module_a(cgb, x, 32_p);
  check_shape(x, 256_p, 35_p, 35_p);

  // Mixed_5c
  x = create_inception_module_a(cgb, x, 64_p);
  check_shape(x, 288_p, 35_p, 35_p);

  // Mixed_5d
  x = create_inception_module_a(cgb, x, 64_p);
  check_shape(x, 288_p, 35_p, 35_p);

  // Mixed_6a
  x = create_inception_module_b(cgb, x);
  check_shape(x, 768_p, 17_p, 17_p);

  // Mixed_6b
  x = create_inception_module_c(cgb, check_shape, x, 128_p);
  check_shape(x, 768_p, 17_p, 17_p);

  // Mixed_6c
  x = create_inception_module_c(cgb, check_shape, x, 160_p);
  check_shape(x, 768_p, 17_p, 17_p);

  // Mixed_6d
  x = create_inception_module_c(cgb, check_shape, x, 160_p);
  check_shape(x, 768_p, 17_p, 17_p);

  // Mixed_6e
  x = create_inception_module_c(cgb, check_shape, x, 192_p);
  check_shape(x, 768_p, 17_p, 17_p);

  std::optional<tensor_guid_t> aux;
  if (config.aux_logits) {
    aux = create_inception_aux(cgb, check_shape, x, config.num_classes);
    check_shape(aux.value(), config.num_classes);
  }

  // Mixed_7a
  x = create_inception_module_d(cgb, x);
  check_shape(x, 1280_p, 8_p, 8_p);

  // Mixed_7b
  x = create_inception_module_e(cgb, x);
  check_shape(x, 2048_p, 8_p, 8_p);

  // Mixed_7c
  x = create_inception_module_e(cgb, x);
  check_shape(x, 2048_p, 8_p, 8_p);

  x = create_final_layers(cgb, check_shape, x, config.num_classes);
  check_shape(x, config.num_classes);

  return InceptionV3Output{
      x,
      aux,
  };
}

ComputationGraph
    get_inception_v3_computation_graph(InceptionV3Config const &config) {
  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<positive_int>{
          config.batch_size,
          3_p,
          299_p,
          299_p,
      }},
      DataType::FLOAT,
  };

  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES);
  InceptionV3Output output = create_inception_v3(cgb, config, input);

  return cgb.computation_graph;
}

} // namespace FlexFlow
