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

#include "task-spec/ops/concat.h"
#include "kernels/concat_kernels.h"
#include "task-spec/op_task_signature.h"
#include "task-spec/profiling.h"
#include "task-spec/variadic_tensor_ref.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Concat;

enum Slots {
  INPUTS,
  OUTPUT,
  ATTRS,
  PROFILING,
  HANDLE,
  NUM_INPUTS,
  KERNEL_DEVICE_TYPE
};

OpTaskInvocation forward(ConcatAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind(INPUTS, get_input_tensors());
  binding.bind(OUTPUT, output_tensor(0_n));
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return OpTaskInvocation{
      task_id_t::CONCAT_FWD_TASK_ID,
      binding,
  };
}

OpTaskInvocation backward(ConcatAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::CONCAT_BWD_TASK_ID,
      b,
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto const &attrs = acc.get_argument<ConcatAttrs>(ATTRS);

  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto inputs = acc.get_variadic_tensor<Permissions::RO>(INPUTS);

  assert(inputs.size() <= MAX_NUM_INPUTS);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Concat] forward_time = {:.2lf}ms\n",
                 output,
                 inputs,
                 attrs.axis);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto const &attrs = acc.get_argument<ConcatAttrs>(ATTRS);

  auto input_grads = acc.get_variadic_tensor_grad<Permissions::RW>(INPUTS);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  assert(input_grads.size() <= MAX_NUM_INPUTS);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Concat] backward_time = {:.2lf}ms\n",
                 output_grad,
                 input_grads,
                 attrs.axis);
}

TaskImplFunction get_concat_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_concat_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_concat_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ConcatAttrs>(ATTRS);
  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_input_slot(INPUTS, SlotType::VARIADIC);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

OpTaskSignature get_concat_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_concat_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(ConcatAttrs const &) {
  return {task_id_t::CONCAT_FWD_TASK_ID, task_id_t::CONCAT_BWD_TASK_ID};
}

}; // namespace FlexFlow
