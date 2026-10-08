/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/passes/SwiGLU.h>

#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

void swiglu_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // et_vk.swiglu.default args: [gate, up, out]
  const int gate_id = args.at(0);
  const int up_id = args.at(1);
  const int out_id = args.at(2);

  const auto& gate = graph.get_tensor(gate_id);
  const auto& up = graph.get_tensor(up_id);
  const auto& out = graph.get_tensor(out_id);
  if (!utils::is_fp32_tensor(gate) || !utils::is_fp32_tensor(up) ||
      !utils::is_fp32_tensor(out)) {
    throw std::runtime_error("swiglu: only fp32 is supported");
  }
  if (gate.dims != up.dims || gate.dims != out.dims) {
    throw std::runtime_error("swiglu: gate, up, and out shapes must match");
  }

  passes::add_silu_mul_fused_dispatch(
      graph, {gate_id, up_id}, gate_id, up_id, out_id);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.swiglu.default, swiglu_impl);
}

} // namespace executorch::backends::webgpu
