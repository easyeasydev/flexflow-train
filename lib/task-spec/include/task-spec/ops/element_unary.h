#ifndef _ELEMENT_UNARY_H
#define _ELEMENT_UNARY_H

#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ElementUnaryAttrs const &);

TaskImplFunction get_element_unary_init_task_impl();
TaskImplFunction get_element_unary_fwd_task_impl();
TaskImplFunction get_element_unary_bwd_task_impl();

OpTaskSignature get_element_unary_init_signature();
OpTaskSignature get_element_unary_fwd_signature();
OpTaskSignature get_element_unary_bwd_signature();

OpTaskInvocation init(ElementUnaryAttrs const &);
OpTaskInvocation forward(ElementUnaryAttrs const &);
OpTaskInvocation backward(ElementUnaryAttrs const &);

} // namespace FlexFlow

#endif
