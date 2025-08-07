#include "kernels/dropout_kernels_cpu.h"
#include "utils/exception.h"
#include <torch/nn/modules/dropout.h>
#include <torch/torch.h>

namespace FlexFlow::Kernels::Dropout {

void cpu_forward_kernel(float const *input_ptr, float *output_ptr) {

  torch::nn::DropoutOptions options{};
  // TODO: find out how to get the dropout config
  options.p(0.42).inplace(true);

  torch::nn::Dropout torch_dropout(options);

  // TODO: convert input_ptr to torch::Tensor
  torch::Tensor torch_input{};

  torch::Tensor torch_output = torch_dropout->forward(torch_input);

  torch_output.backward();

  // TODO: convert torch::Tensor to output_ptr

  // NOT_IMPLEMENTED();
}

void cpu_backward_kernel(float const *output_grad_ptr, float *input_grad_ptr) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Dropout
