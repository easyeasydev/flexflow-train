/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "internal/device.h"
#include "kernels/dropout_kernels.h"
#include "kernels/ff_handle.h"

namespace FlexFlow {
namespace Kernels {
namespace Dropout {

DropoutPerDeviceState gpu_init_kernel(PerDeviceFFHandle const &handle,
                                      float rate,
                                      unsigned long long seed,
                                      TensorShape const &output_shape,
                                      Allocator &allocator) {
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffDropoutDescriptor_t dropoutDesc;
  void *reserveSpace;
  void *dropoutStates;
  size_t reserveSpaceSize;
  size_t dropoutStateSize;
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
  checkCUDNN(cudnnDropoutGetStatesSize(handle.dnn, &(dropoutStateSize)));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(inputTensor, output_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(outputTensor, output_shape));
  checkCUDNN(
      cudnnDropoutGetReserveSpaceSize(outputTensor, &(reserveSpaceSize)));
  {
    // allocate memory for dropoutStates and reserveSpace
    size_t totalSize = dropoutStateSize + reserveSpaceSize;
    dropoutStates = allocator.allocate(totalSize);
    reserveSpace = ((char *)dropoutStates) + dropoutStateSize;
  }
  checkCUDNN(cudnnSetDropoutDescriptor(
      dropoutDesc, handle.dnn, rate, dropoutStates, dropoutStateSize, seed));
  DropoutPerDeviceState per_device_state = DropoutPerDeviceState{
      /*handle=*/handle,
      /*inputTensor=*/inputTensor,
      /*outputTensor=*/outputTensor,
      /*dropoutDesc=*/dropoutDesc,
      /*reserveSpace=*/reserveSpace,
      /*dropoutStates=*/dropoutStates,
      /*reserveSpaceSize=*/reserveSpaceSize,
      /*dropoutStateSize=*/dropoutStateSize,
  };
  return per_device_state;
}

void gpu_forward_kernel(cudaStream_t stream,
                        DropoutPerDeviceState const &m,
                        float const *input_ptr,
                        float *output_ptr) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  checkCUDNN(cudnnDropoutForward(m.handle.dnn,
                                 m.dropoutDesc,
                                 m.inputTensor,
                                 input_ptr,
                                 m.outputTensor,
                                 output_ptr,
                                 m.reserveSpace,
                                 m.reserveSpaceSize));
}

void gpu_backward_kernel(cudaStream_t stream,
                         DropoutPerDeviceState const &m,
                         float const *output_grad_ptr,
                         float *input_grad_ptr) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  checkCUDNN(cudnnDropoutBackward(m.handle.dnn,
                                  m.dropoutDesc,
                                  m.outputTensor,
                                  output_grad_ptr,
                                  m.inputTensor,
                                  input_grad_ptr,
                                  m.reserveSpace,
                                  m.reserveSpaceSize));
}

void gpu_cleanup_kernel(Allocator &allocator,
                        DropoutPerDeviceState const &per_device_state) {
  allocator.deallocate(per_device_state.dropoutStates);
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.outputTensor));
  checkCUDNN(cudnnDestroyDropoutDescriptor(per_device_state.dropoutDesc));
}

} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow
