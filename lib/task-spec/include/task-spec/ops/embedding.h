#ifndef _FLEXFLOW_EMBEDDING_H
#define _FLEXFLOW_EMBEDDING_H

#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(EmbeddingAttrs const &);

TaskImplFunction get_embedding_fwd_task_impl();
TaskImplFunction get_embedding_bwd_task_impl();

OpTaskSignature get_embedding_fwd_signature();
OpTaskSignature get_embedding_bwd_signature();

OpTaskInvocation forward(EmbeddingAttrs const &);
OpTaskInvocation backward(EmbeddingAttrs const &);

} // namespace FlexFlow

#endif
