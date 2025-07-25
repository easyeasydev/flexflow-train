#include "local-execution/local_task_argument_accessor.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/transform.h"
#include "utils/hash/pair.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalTaskArgumentAccessor::LocalTaskArgumentAccessor(
    Allocator const &allocator,
    std::unordered_map<tensor_sub_slot_id_t, TensorSlotBacking> const
        &tensor_slots_backing,
    std::unordered_map<slot_id_t, ConcreteArgSpec> const &arg_slots_backing)
    : allocator(allocator), tensor_slots_backing(tensor_slots_backing),
      arg_slots_backing(arg_slots_backing){};

ConcreteArgSpec const &
    LocalTaskArgumentAccessor::get_concrete_arg(slot_id_t name) const {
  return this->arg_slots_backing.at(name);
}

GenericTensorAccessor LocalTaskArgumentAccessor::get_tensor(
    slot_id_t slot, Permissions priv, TensorType tensor_type) const {
  tensor_sub_slot_id_t slot_tensor_type =
      tensor_sub_slot_id_t{slot, tensor_type};
  GenericTensorAccessorW tensor_backing =
      this->tensor_slots_backing.at(slot_tensor_type).require_single();
  if (priv == Permissions::RO) {
    GenericTensorAccessorR readonly_tensor_backing =
        read_only_accessor_from_write_accessor(tensor_backing);
    return readonly_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing;
  } else {
    PANIC(fmt::format("Unhandled privilege mode {}", priv));
  }
}

VariadicGenericTensorAccessor LocalTaskArgumentAccessor::get_variadic_tensor(
    slot_id_t slot, Permissions priv, TensorType tensor_type) const {
  tensor_sub_slot_id_t slot_tensor_type =
      tensor_sub_slot_id_t{slot, tensor_type};
  std::vector<GenericTensorAccessorW> variadic_tensor_backing =
      this->tensor_slots_backing.at(slot_tensor_type).require_variadic();
  if (priv == Permissions::RO) {
    std::vector<GenericTensorAccessorR> readonly_variadic_tensor_backing = {};
    for (GenericTensorAccessorW const &tensor_backing :
         variadic_tensor_backing) {
      readonly_variadic_tensor_backing.push_back(
          read_only_accessor_from_write_accessor(tensor_backing));
    }
    return readonly_variadic_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return variadic_tensor_backing;
  } else {
    PANIC(fmt::format("Unhandled privilege mode {}", priv));
  }
}

Allocator LocalTaskArgumentAccessor::get_allocator() const {
  return this->allocator;
}

size_t LocalTaskArgumentAccessor::get_device_idx() const {
  return 0;
}

} // namespace FlexFlow
