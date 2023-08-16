// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <c10/macros/Export.h>
#include <ATen/core/ivalue.h>

namespace torch {
namespace aot_inductor {

class TORCH_API ProxyExecutor : public torch::CustomClassHolder {
 public:
  ProxyExecutor() {}
  virtual ~ProxyExecutor() {}

  virtual void call_function(
    const std::string& node_name,
    int64_t* flatten_int_args,
    void** flatten_args_outputs) = 0;
};


} // namespace aot_inductor
} // namespace torch
