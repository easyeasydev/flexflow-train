#ifndef _FLEXFLOW_REVERSE_H_
#define _FLEXFLOW_REVERSE_H_

#include "op-attrs/ops/reverse_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ReverseAttrs const &);

TaskImplFunction get_reverse_fwd_task_impl();
TaskImplFunction get_reverse_bwd_task_impl();

OpTaskSignature get_reverse_fwd_signature();
OpTaskSignature get_reverse_bwd_signature();

OpTaskInvocation forward(ReverseAttrs const &);
OpTaskInvocation backward(ReverseAttrs const &);

} // namespace FlexFlow

#endif
