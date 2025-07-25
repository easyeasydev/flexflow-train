#include "kernels/element_unary_kernels.h"
#include "kernels/element_unary_kernels_cpu.h"
#include "kernels/element_unary_kernels_gpu.h"

namespace FlexFlow::Kernels::ElementUnary {

std::optional<ElementUnaryPerDeviceState>
    init_kernel(DeviceType device_type,
                TensorShape const &input_shape,
                TensorShape const &output_shape,
                ElementUnaryAttrs const &attrs) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
        /*input_shape=*/input_shape,
        /*output_shape=*/output_shape,
        /*attrs=*/attrs);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
}

void forward_kernel(
    device_stream_t const &stream,
    std::optional<ElementUnaryPerDeviceState> const &per_device_state,
    ElementUnaryAttrs const &attrs,
    device_handle_t const &handle,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*attrs=*/attrs,
        /*handle=*/handle.require_for_gpu(),
        /*input=*/input,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    ASSERT(handle.is_for_cpu());
    cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);
  }
}

void backward_kernel(
    device_stream_t const &stream,
    std::optional<ElementUnaryPerDeviceState> const &per_device_state,
    ElementUnaryAttrs const &attrs,
    device_handle_t const &handle,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*attrs=*/attrs,
        /*handle=*/handle.require_for_gpu(),
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    ASSERT(handle.is_for_cpu());
    cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad);
  }
}

void cleanup_kernel(
    DeviceType device_type,
    std::optional<ElementUnaryPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow::Kernels::ElementUnary
