#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_LOSS_FUNCTION_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_LOSS_FUNCTION_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow {

void sparse_categorical_crossentropy_loss_backward_kernel(
    device_stream_t const &stream,
    float *logit_grad_ptr,
    float const *logit_ptr,
    int const *label_ptr,
    size_t logit_volume,
    size_t logit_grad_volume,
    int num_samples,
    int num_classes,
    int k,
    float scale_factor);

void categorical_crossentropy_loss_backward_kernel(
    device_stream_t const &stream,
    GenericTensorAccessorW const &logit_grad,
    GenericTensorAccessorR const &logit,
    GenericTensorAccessorR const &label,
    float scale_factor);

void mean_squared_error_avg_loss_backward_kernel(device_stream_t const &stream,
                                                 float *logit_grad_ptr,
                                                 float const *logit_ptr,
                                                 float const *label_ptr,
                                                 size_t logit_volume,
                                                 size_t logit_grad_volume,
                                                 float scale_factor);

void identity_loss_backward_kernel(device_stream_t const &stream,
                                   float *loss_grad_ptr,
                                   float const *loss_ptr,
                                   size_t loss_volume,
                                   size_t loss_grad_volume,
                                   float csale_factor);

} // namespace FlexFlow

#endif
