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
#include "kernels/split_kernels_gpu.h"

namespace FlexFlow {

namespace Kernels {
namespace Split {

void gpu_forward_kernel(cudaStream_t stream,
                        float **out_ptrs,
                        float const *in_ptr,
                        int const *out_blk_sizes,
                        int in_blk_size,
                        int num_blks,
                        int numOutputs) {

  for (int i = 0; i < numOutputs; i++) {
    copy_with_stride<<<GET_BLOCKS(out_blk_sizes[i] * num_blks),
                       CUDA_NUM_THREADS,
                       0,
                       stream>>>(
        out_ptrs[i], in_ptr, num_blks, out_blk_sizes[i], in_blk_size);
    in_ptr += out_blk_sizes[i];
  }
}

void gpu_backward_kernel(cudaStream_t stream,
                         float *in_grad_ptr,
                         float const **out_grad_ptr,
                         int const *out_blk_sizes,
                         int in_blk_size,
                         int num_blks,
                         int numOutputs) {

  for (int i = 0; i < numOutputs; i++) {
    add_with_stride<<<GET_BLOCKS(out_blk_sizes[i] * num_blks),
                      CUDA_NUM_THREADS,
                      0,
                      stream>>>(
        in_grad_ptr, out_grad_ptr[i], num_blks, in_blk_size, out_blk_sizes[i]);
    in_grad_ptr += out_blk_sizes[i];
  }
  // checkCUDA(cudaDeviceSynchronize());
}

} // namespace Split
} // namespace Kernels
} // namespace FlexFlow
