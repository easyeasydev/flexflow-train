#include "op-attrs/ops/concat.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(ConcatAttrs, std::vector<TensorShape>)") {
    ConcatAttrs attrs = ConcatAttrs{
        ff_dim_t{1_n},
    };

    SUBCASE("empty input shapes list passed") {
      std::vector<TensorShape> input_shapes = {};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input_shapes));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    positive_int dim0_size = 12_p;
    positive_int dim2_size = 20_p;
    TensorShape input_shape1 = TensorShape{
        TensorDims{FFOrdered{
            dim0_size,
            14_p,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    SUBCASE("single element input shapes list passed") {
      std::vector<TensorShape> input_shapes = {input_shape1};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input_shapes));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    TensorShape input_shape2 = TensorShape{
        TensorDims{FFOrdered{
            dim0_size,
            16_p,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    TensorShape input_shape3 = TensorShape{
        TensorDims{FFOrdered{dim0_size, 18_p, dim2_size}},
        DataType::FLOAT,
    };

    SUBCASE("input shapes do not shared the same num_dims") {
      TensorShape mismatched_num_dims = TensorShape{
          TensorDims{FFOrdered{
              dim0_size,
              20_p,
              dim2_size,
              1_p,
          }},
          DataType::FLOAT,
      };

      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3, mismatched_num_dims};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input_shapes));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("concat axis is out of bounds") {
      attrs = ConcatAttrs{
          ff_dim_t{3_n},
      };

      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3};

      std::optional<TensorShape> result =
          optional_from_expected(get_output_shape(attrs, input_shapes));
      std::optional<TensorShape> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input shapes are valid") {
      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3};

      tl::expected<TensorShape, std::string> result =
          get_output_shape(attrs, input_shapes);
      tl::expected<TensorShape, std::string> correct = TensorShape{
          TensorDims{FFOrdered{
              dim0_size,
              14_p + 16_p + 18_p,
              dim2_size,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_output_shape(ConcatAttrs, std::vector<ParallelTensorShape>)") {
    ConcatAttrs attrs = ConcatAttrs{
        ff_dim_t{1_n},
    };

    positive_int dim0_size = 12_p;
    positive_int dim2_size = 20_p;

    TensorShape input_shape1 = TensorShape{
        TensorDims{FFOrdered{
            dim0_size,
            14_p,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    TensorShape input_shape2 = TensorShape{
        TensorDims{FFOrdered{
            dim0_size,
            16_p,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    TensorShape input_shape3 = TensorShape{
        TensorDims{FFOrdered{dim0_size, 18_p, dim2_size}},
        DataType::FLOAT,
    };

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{dim0_size, 14_p + 16_p + 18_p, dim2_size}},
        DataType::FLOAT,
    };

    auto lift_input1 = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2) {
      return lift_to_parallel_with_degrees(
          input_shape1, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    auto lift_input2 = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2) {
      return lift_to_parallel_with_degrees(
          input_shape2, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    auto lift_input3 = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2) {
      return lift_to_parallel_with_degrees(
          input_shape3, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    auto lift_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2) {
      return lift_to_parallel_with_degrees(
          output_shape, o_sum, o_eq, FFOrdered{o0, o1, o2});
    };

    SUBCASE("sum reduction parallelism") {
      SUBCASE("matching") {
        SumDegree sum_degree = SumDegree{2_p};

        std::vector<ParallelTensorShape> inputs = {
            lift_input1(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
            lift_input2(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
            lift_input3(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
        };

        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, inputs);
        tl::expected<ParallelTensorShape, std::string> correct =
            lift_output(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{2_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
            lift_input2(SumDegree{4_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
            lift_input3(SumDegree{4_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("discard copy reduction parallelism") {
      SUBCASE("matching") {
        DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p),
            lift_input2(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p),
            lift_input3(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p),
        };

        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, inputs);
        tl::expected<ParallelTensorShape, std::string> correct =
            lift_output(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1_p}, DiscardCopyDegree{2_p}, 1_p, 1_p, 1_p),
            lift_input2(SumDegree{1_p}, DiscardCopyDegree{2_p}, 1_p, 1_p, 1_p),
            lift_input3(SumDegree{1_p}, DiscardCopyDegree{4_p}, 1_p, 1_p, 1_p),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallelism in axis dim") {
      SUBCASE("matching") {
        positive_int degree = 2_p;

        std::vector<ParallelTensorShape> inputs = {
            lift_input1(
                SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree, 1_p),
            lift_input2(
                SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree, 1_p),
            lift_input3(
                SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree, 1_p),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
            lift_input2(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p),
            lift_input3(SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, 2_p, 1_p),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallelism in non-axis shard dims") {
      SUBCASE("matching") {
        positive_int degree0 = 2_p;
        positive_int degree2 = 4_p;

        std::vector<ParallelTensorShape> inputs = {
            lift_input1(
                SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2),
            lift_input2(
                SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2),
            lift_input3(
                SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2),
        };

        tl::expected<ParallelTensorShape, std::string> result =
            get_output_shape(attrs, inputs);
        tl::expected<ParallelTensorShape, std::string> correct = lift_output(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorShape> inputs = {
            lift_input1(SumDegree{1_p}, DiscardCopyDegree{1_p}, 2_p, 1_p, 4_p),
            lift_input2(SumDegree{1_p}, DiscardCopyDegree{1_p}, 4_p, 1_p, 2_p),
            lift_input3(SumDegree{1_p}, DiscardCopyDegree{1_p}, 4_p, 1_p, 2_p),
        };

        std::optional<ParallelTensorShape> result =
            optional_from_expected(get_output_shape(attrs, inputs));
        std::optional<ParallelTensorShape> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("parallelism degrees are not mutually exclusive") {
      SumDegree sum_degree = SumDegree{3_p};
      DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{5_p};
      positive_int degree0 = 2_p;
      positive_int degree2 = 4_p;

      std::vector<ParallelTensorShape> inputs = {
          lift_input1(sum_degree, discard_copy_degree, degree0, 1_p, degree2),
          lift_input2(sum_degree, discard_copy_degree, degree0, 1_p, degree2),
          lift_input3(sum_degree, discard_copy_degree, degree0, 1_p, degree2),
      };

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, inputs);
      tl::expected<ParallelTensorShape, std::string> correct =
          lift_output(sum_degree, discard_copy_degree, degree0, 1_p, degree2);

      CHECK(result == correct);
    }
  }
}
