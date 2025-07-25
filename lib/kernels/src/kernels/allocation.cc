#include "kernels/allocation.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

void *Allocator::allocate(size_t mem_size) {
  return this->i_allocator->allocate(mem_size);
}

void Allocator::deallocate(void *ptr) {
  this->i_allocator->deallocate(ptr);
}

DeviceType Allocator::get_allocation_device_type() const {
  return this->i_allocator->get_allocation_device_type();
}

GenericTensorAccessorW
    Allocator::allocate_tensor(TensorShape const &tensor_shape) {
  void *ptr = this->allocate(
      get_size_in_bytes(tensor_shape).unwrap_num_bytes().unwrap_nonnegative());
  return GenericTensorAccessorW{
      tensor_shape,
      ptr,
      this->get_allocation_device_type(),
  };
}

void Allocator::deallocate_tensor(GenericTensorAccessorW const &t) {
  this->deallocate(t.ptr);
}

void Allocator::deallocate_tensor(GenericTensorAccessorR const &t) {
  this->deallocate(const_cast<void *>(t.ptr));
}

} // namespace FlexFlow
