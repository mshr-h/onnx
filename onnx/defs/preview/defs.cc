/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>
#include <functional>
#include <unordered_set>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

static constexpr const char* FlexAttention_ver1_doc = R"DOC(
Computes scaled dot-product attention over rank-4 (batched, multi-head) inputs,
with optional user-provided customization subgraphs at up to three stages:
  (1) score_mod: modify each scalar attention score after Q·K^T
  (2) mask_mod : determine which (q_idx, k_idx) connections are allowed
  (3) prob_mod : modify each scalar probability after Softmax

Inputs MUST be rank-4 tensors with shapes:
  Q: (batch_size, q_num_heads, q_sequence_length, head_size)
  K: (batch_size, kv_num_heads, kv_sequence_length, head_size)
  V: (batch_size, kv_num_heads, kv_sequence_length, v_head_size)

The output has shape:
  Y: (batch_size, q_num_heads, q_sequence_length, v_head_size)
)DOC";

// Find the last node index that produces any of the given value-names.
static int FindLastProducerIndex(const FunctionProto& fp, const std::vector<std::string>& values) {
  std::unordered_set<std::string> set(values.begin(), values.end());
  int last = -1;
  for (int i = 0; i < fp.node_size(); ++i) {
    for (const auto& out : fp.node(i).output()) {
      if (set.count(out)) {
        last = std::max(last, i);
        break;
      }
    }
  }
  return last;
}

// Insert a node at position `index` in functionProto.node().
static void InsertNodeAt(FunctionProto& fp, const NodeProto& n, int index) {
  auto* nodes = fp.mutable_node();
  nodes->Add()->CopyFrom(n); // append
  int cur = nodes->size() - 1;
  while (cur > index) {
    nodes->SwapElements(cur, cur - 1);
    --cur;
  }
}

static void RemapGraphProtoNames(
    GraphProto* g,
    const std::function<std::string(const std::string&)>& map_name);

static void RemapNodeProtoNames(
    NodeProto* n,
    const std::function<std::string(const std::string&)>& map_name) {
  // Remap inputs/outputs
  const auto old_inputs = n->input();
  const auto old_outputs = n->output();
  n->clear_input();
  n->clear_output();
  for (const auto& in : old_inputs)
    n->add_input(map_name(in));
  for (const auto& out : old_outputs)
    n->add_output(map_name(out));

  // Recursively remap attribute graphs (If/Loop/Scan/etc.).
  for (int i = 0; i < n->attribute_size(); ++i) {
    auto* attr = n->mutable_attribute(i);
    if (attr->has_g()) {
      RemapGraphProtoNames(attr->mutable_g(), map_name);
    }
    if (attr->graphs_size() > 0) {
      for (int j = 0; j < attr->graphs_size(); ++j) {
        RemapGraphProtoNames(attr->mutable_graphs(j), map_name);
      }
    }
  }
}

static void RemapGraphProtoNames(
    GraphProto* g,
    const std::function<std::string(const std::string&)>& map_name) {
  // Graph inputs/outputs/value_info
  for (int i = 0; i < g->input_size(); ++i) {
    g->mutable_input(i)->set_name(map_name(g->input(i).name()));
  }
  for (int i = 0; i < g->output_size(); ++i) {
    g->mutable_output(i)->set_name(map_name(g->output(i).name()));
  }
  for (int i = 0; i < g->value_info_size(); ++i) {
    g->mutable_value_info(i)->set_name(map_name(g->value_info(i).name()));
  }

  // Initializers
  for (int i = 0; i < g->initializer_size(); ++i) {
    g->mutable_initializer(i)->set_name(map_name(g->initializer(i).name()));
  }
  for (int i = 0; i < g->sparse_initializer_size(); ++i) {
    auto* st = g->mutable_sparse_initializer(i);
    // SparseTensorProto has no top-level "name"; the TensorProto "values" name serves as the
    // initializer name when used in GraphProto::sparse_initializer.
    if (st->has_values()) {
      st->mutable_values()->set_name(map_name(st->values().name()));
    }
    if (st->has_indices()) {
      st->mutable_indices()->set_name(map_name(st->indices().name()));
    }
  }

  // Nodes
  for (int i = 0; i < g->node_size(); ++i) {
    RemapNodeProtoNames(g->mutable_node(i), map_name);
  }
}

// Minimal inliner for GraphProto (attribute graphs) into another GraphProto.
// - Maps subgraph input/output value-names to caller-provided names using io_map.
// - Prefixes all other value names (node outputs, initializers, intermediate values) with `prefix`
//   to avoid collisions.
// Notes:
// - Supports nested subgraphs inside node attributes (If/Loop/Scan/etc.) by recursively remapping
//   AttributeProto::g/graphs GraphProto value-names.
// - Copies initializers; does not copy value_info (not required for execution).
static void InlineGraphInto(
    GraphProto* dst,
    const GraphProto& src,
    const std::unordered_map<std::string, std::string>& io_map,
    const std::string& prefix) {
  auto map_name = [&](const std::string& n) -> std::string {
    if (n.empty())
      return n;
    auto it = io_map.find(n);
    if (it != io_map.end())
      return it->second;
    return prefix + n;
  };

  // Copy initializers
  for (const auto& init : src.initializer()) {
    TensorProto* ni = dst->add_initializer();
    *ni = init;
    ni->set_name(map_name(init.name()));
  }

  // Copy sparse initializers
  for (const auto& s_init : src.sparse_initializer()) {
    SparseTensorProto* ns = dst->add_sparse_initializer();
    *ns = s_init;
    // SparseTensorProto has no top-level "name"; TensorProto "values" name serves as the
    // initializer name when used in GraphProto::sparse_initializer.
    if (ns->has_values()) {
      ns->mutable_values()->set_name(map_name(ns->values().name()));
    }
    if (ns->has_indices()) {
      ns->mutable_indices()->set_name(map_name(ns->indices().name()));
    }
  }

  // Copy nodes, remapping inputs/outputs
  for (const auto& n : src.node()) {
    NodeProto* nn = dst->add_node();
    *nn = n;
    RemapNodeProtoNames(nn, map_name);
  }
}

static void ValidateFlexAttentionModGraph(
    InferenceContext& ctx,
    const AttributeProto* attr,
    size_t expected_inputs,
    const std::string& attr_name,
    std::optional<int32_t> expected_output_elem_type,
    bool require_scalar_output) {
  if (!attr)
    return;
  if (!attr->has_g()) {
    fail_shape_inference("Attribute ", attr_name, " must be a graph.");
  }

  const auto& g = attr->g();
  if (g.input_size() != static_cast<int>(expected_inputs)) {
    fail_shape_inference(
        "Attribute ", attr_name, " expected ", expected_inputs, " inputs but graph has ", g.input_size(), ".");
  }

  if (g.output_size() != 1) {
    fail_shape_inference("Attribute ", attr_name, " must have exactly one output.");
  }

  auto check_scalar_tensor_if_typed = [&](const ValueInfoProto& vi, const char* what) {
    if (!vi.has_type())
      return;
    const auto& tp = vi.type();
    if (!tp.has_tensor_type()) {
      fail_shape_inference("Attribute ", attr_name, " ", what, " must be a tensor type if typed.");
    }
    const auto& tt = tp.tensor_type();
    if (tt.has_shape() && tt.shape().dim_size() != 0) {
      fail_shape_inference("Attribute ", attr_name, " ", what, " must be a scalar tensor.");
    }
  };

  auto check_elem_type_if_typed = [&](const ValueInfoProto& vi, int32_t et, const char* what) {
    if (!vi.has_type())
      return;
    const auto& tp = vi.type();
    if (!tp.has_tensor_type())
      return;
    const auto& tt = tp.tensor_type();
    if (tt.elem_type() != et) {
      fail_shape_inference(
          "Attribute ", attr_name, " ", what, " element type mismatch. Expected ", et, ", got ", tt.elem_type(), ".");
    }
  };

  const bool is_mask_mod = (expected_inputs == 4);

  // Inputs: enforce scalar-ness if typed.
  for (int i = 0; i < g.input_size(); ++i) {
    check_scalar_tensor_if_typed(g.input(i), ("input(" + std::to_string(i) + ")").c_str());
  }

  // Typed element-type checks (best-effort):
  // - score_mod/prob_mod: input(0) and output are softmax_precision
  // - mask_mod: output is bool
  if (expected_output_elem_type.has_value()) {
    if (is_mask_mod) {
      // mask_mod indices are int64
      for (int i = 0; i < 4 && i < g.input_size(); ++i) {
        check_elem_type_if_typed(g.input(i), TensorProto::INT64, ("input(" + std::to_string(i) + ")").c_str());
      }
    } else {
      // score/prob first input
      if (expected_inputs >= 5 && g.input_size() >= 1) {
        check_elem_type_if_typed(g.input(0), expected_output_elem_type.value(), "input(0)");
      }
      // indices (batch, head, q_idx, k_idx) are int64 for score_mod/prob_mod
      if (expected_inputs >= 5) {
        for (int i = 1; i < static_cast<int>(expected_inputs) && i < g.input_size(); ++i) {
          check_elem_type_if_typed(g.input(i), TensorProto::INT64, ("input(" + std::to_string(i) + ")").c_str());
        }
      }
    }
  } else {
    // mask_mod indices are int64
    if (expected_inputs == 4) {
      for (int i = 0; i < 4 && i < g.input_size(); ++i) {
        check_elem_type_if_typed(g.input(i), TensorProto::INT64, ("input(" + std::to_string(i) + ")").c_str());
      }
    }
  }

  if (g.output_size() == 1 && g.output(0).has_type()) {
    const auto& type = g.output(0).type();
    if (!type.has_tensor_type()) {
      fail_shape_inference("Attribute ", attr_name, " output must be a tensor.");
    }
    const auto& tensor_type = type.tensor_type();
    if (expected_output_elem_type.has_value() && tensor_type.elem_type() != expected_output_elem_type.value()) {
      fail_shape_inference(
          "Attribute ",
          attr_name,
          " output element type does not match expected type. Expected ",
          expected_output_elem_type.value(),
          ", got ",
          tensor_type.elem_type(),
          ".");
    }
    if (require_scalar_output && tensor_type.has_shape() && tensor_type.shape().dim_size() != 0) {
      fail_shape_inference("Attribute ", attr_name, " output must be a scalar tensor.");
    }
  }
}

static void FlexAttentionShapeInference(InferenceContext& ctx) {
  if (ctx.getNumInputs() < 3) {
    fail_type_inference("FlexAttention requires query, key, and value inputs.");
  }

  const auto* q_type = ctx.getInputType(0);
  const auto* k_type = ctx.getInputType(1);
  const auto* v_type = ctx.getInputType(2);
  if (!q_type || !k_type || !v_type) {
    return;
  }
  if (!q_type->has_tensor_type() || !k_type->has_tensor_type() || !v_type->has_tensor_type()) {
    fail_type_inference("Inputs 0, 1, and 2 are required tensor types.");
  }

  const auto q_elem_type = q_type->tensor_type().elem_type();
  const auto k_elem_type = k_type->tensor_type().elem_type();
  const auto v_elem_type = v_type->tensor_type().elem_type();
  if (q_elem_type != k_elem_type || q_elem_type != v_elem_type) {
    fail_type_inference("Inputs query, key, and value must have the same element type.");
  }

  auto* output_type = ctx.getOutputType(0)->mutable_tensor_type();
  output_type->set_elem_type(q_elem_type);

  const auto& q_shape = q_type->tensor_type().shape();
  const auto& k_shape = k_type->tensor_type().shape();
  const auto& v_shape = v_type->tensor_type().shape();
  if (q_shape.dim_size() != 4 || k_shape.dim_size() != 4 || v_shape.dim_size() != 4) {
    fail_shape_inference(
        "FlexAttention requires rank-4 inputs: "
        "Q (B, Hq, L, Dqk), K (B, Hkv, S, Dqk), V (B, Hkv, S, Dv).");
  }

  // Validate head sizes between K and V.
  if (k_shape.dim(1).has_dim_value() && v_shape.dim(1).has_dim_value() &&
      k_shape.dim(1).dim_value() != v_shape.dim(1).dim_value()) {
    fail_shape_inference("Key and value must share the same head dimension.");
  }

  // Validate sequence length alignment between K and V.
  if (k_shape.dim(2).has_dim_value() && v_shape.dim(2).has_dim_value() &&
      k_shape.dim(2).dim_value() != v_shape.dim(2).dim_value()) {
    fail_shape_inference("Key and value must share the same sequence length.");
  }

  // Validate depth alignment between Q and K.
  if (q_shape.dim(3).has_dim_value() && k_shape.dim(3).has_dim_value() &&
      q_shape.dim(3).dim_value() != k_shape.dim(3).dim_value()) {
    fail_shape_inference("Query and key must share the same embedding dimension.");
  }

  // Validate enable_gqa attribute.
  const auto enable_gqa = getAttribute(ctx, "enable_gqa", 0);
  if (enable_gqa == 0) {
    if (q_shape.dim(1).has_dim_value() && k_shape.dim(1).has_dim_value() &&
        q_shape.dim(1).dim_value() != k_shape.dim(1).dim_value()) {
      fail_shape_inference("enable_gqa=0 requires Hq == Hkv.");
    }
  } else {
    if (q_shape.dim(1).has_dim_value() && k_shape.dim(1).has_dim_value()) {
      const auto hq = q_shape.dim(1).dim_value();
      const auto hkv = k_shape.dim(1).dim_value();
      if (hkv <= 0 || (hq % hkv) != 0) {
        fail_shape_inference("enable_gqa=1 requires Hq to be divisible by Hkv.");
      }
    }
  }

  auto* output_shape = output_type->mutable_shape();
  output_shape->clear_dim();
  *output_shape->add_dim() = q_shape.dim(0);
  *output_shape->add_dim() = q_shape.dim(1);
  *output_shape->add_dim() = q_shape.dim(2);
  *output_shape->add_dim() = v_shape.dim(3);

  mergeInDimensionInfo(k_shape.dim(0), *output_shape->mutable_dim(0), 0);
  mergeInDimensionInfo(v_shape.dim(0), *output_shape->mutable_dim(0), 0);
  if (enable_gqa == 0) {
    mergeInDimensionInfo(k_shape.dim(1), *output_shape->mutable_dim(1), 1);
  }

  // score_mod/prob_mod run at Softmax precision in the function fallback.
  // PyTorch-like default: for float16/bfloat16 inputs, softmax computations default to float.
  int32_t softmax_elem_type = q_elem_type;
  if (const auto* sp = ctx.getAttribute("softmax_precision")) {
    softmax_elem_type = static_cast<int32_t>(sp->i());
  } else {
    if (softmax_elem_type == TensorProto::FLOAT16 || softmax_elem_type == TensorProto::BFLOAT16) {
      softmax_elem_type = TensorProto::FLOAT;
    }
  }
  // Keep this aligned with builder's allowed set.
  if (softmax_elem_type != TensorProto::FLOAT && softmax_elem_type != TensorProto::FLOAT16 &&
      softmax_elem_type != TensorProto::BFLOAT16 && softmax_elem_type != TensorProto::DOUBLE) {
    fail_type_inference("softmax_precision must be specified when inputs are not float/float16/bfloat16/double.");
  }

  ValidateFlexAttentionModGraph(ctx, ctx.getAttribute("score_mod"), 5, "score_mod", softmax_elem_type, true);
  ValidateFlexAttentionModGraph(ctx, ctx.getAttribute("mask_mod"), 4, "mask_mod", TensorProto::BOOL, true);
  ValidateFlexAttentionModGraph(ctx, ctx.getAttribute("prob_mod"), 5, "prob_mod", softmax_elem_type, true);
}

ONNX_PREVIEW_OPERATOR_SET_SCHEMA(
    FlexAttention,
    1,
    OpSchema()
        .SetDoc(FlexAttention_ver1_doc)
        .Input(
            0,
            "Q",
            "Query tensor. 4D tensor with shape "
            "`(batch_size, q_num_heads, q_sequence_length, head_size)`.",
            "T1")
        .Input(
            1,
            "K",
            "Key tensor. 4D tensor with shape "
            "`(batch_size, kv_num_heads, kv_sequence_length, head_size)`.",
            "T1")
        .Input(
            2,
            "V",
            "Value tensor. 4D tensor with shape "
            "`(batch_size, kv_num_heads, kv_sequence_length, v_head_size)`.",
            "T1")
        .Output(
            0,
            "Y",
            "The output tensor. 4D tensor with shape "
            "`(batch_size, q_num_heads, q_sequence_length, v_head_size)`.",
            "T1")
        .Attr(
            "scale",
            "Scaling factor applied to Q*K^T. Default value is 1/sqrt(head_size) when omitted. "
            "For numerical stability, implementations may scale Q and K by sqrt(scale) before MatMul.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
        .Attr(
            "softmax_precision",
            "The floating-point precision used in softmax computation. "
            "If softmax precision is not provided, the default is: float for float16/bfloat16 inputs, otherwise the "
            "same precision as the input of softmax (Q and K). If Q/K/V are not float/float16/bfloat16/double, "
            "softmax_precision must be specified.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "enable_gqa",
            "If 0: requires Hq == Hkv. If 1: enables Grouped Query Attention; K/V heads are broadcast to query heads.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "mask_value",
            "Value used to replace masked scores before Softmax (approximation of -inf).",
            AttributeProto::FLOAT,
            -3.402823466e+38f)
        .Attr(
            "score_mod",
            "Optional score modifier graph.\n"
            "The graph MUST have exactly 5 inputs in this exact order:\n"
            "  (score, batch, head, q_idx, k_idx)\n"
            "and exactly 1 output (score_out).\n"
            "All five inputs are scalar tensors (0-D). The output is a scalar tensor (0-D).\n"
            "batch/head/q_idx/k_idx are INT64 scalars; score is a scalar of softmax_precision (or default precision when omitted).",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr(
            "mask_mod",
            "Optional mask modifier graph.\n"
            "The graph MUST have exactly 4 inputs in this exact order:\n"
            "  (batch, head, q_idx, k_idx)\n"
            "and exactly 1 output (mask_out).\n"
            "All inputs are INT64 scalar tensors (0-D). mask_out is a BOOL scalar tensor (0-D).",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr(
            "prob_mod",
            "Optional probability modifier graph.\n"
            "The graph MUST have exactly 5 inputs in this exact order:\n"
            "  (prob, batch, head, q_idx, k_idx)\n"
            "and exactly 1 output (prob_out).\n"
            "All five inputs are scalar tensors (0-D). The output is a scalar tensor (0-D).\n"
            "batch/head/q_idx/k_idx are INT64 scalars; prob is a scalar of softmax_precision (or default precision when omitted).",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain Q, K and V inputs types to float tensors.")
        .TypeAndShapeInferenceFunction(FlexAttentionShapeInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          int64_t float_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

          // Get input types
          auto t_qk = ctx.getInputType(0);
          if ((t_qk == nullptr) || (!t_qk->has_tensor_type()))
            return false;
          int64_t T1 = t_qk->tensor_type().elem_type();

          // Determine precision types for Softmax
          auto softmax_precision_attr = ctx.getAttribute("softmax_precision");
          int64_t softmax_precision;
          if (softmax_precision_attr != nullptr) {
            softmax_precision = softmax_precision_attr->i();
          } else {
            // PyTorch-like default: for float16/bfloat16 inputs, compute softmax in float.
            if (T1 == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
                T1 == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
              softmax_precision = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
            } else {
              softmax_precision = T1;
            }
          }
          if ((softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE)) {
            return false;
          }

          const bool enable_gqa =
              (ctx.getAttribute("enable_gqa") != nullptr && ctx.getAttribute("enable_gqa")->i() != 0);

          auto* score_mod_attr = ctx.getAttribute("score_mod");
          auto* mask_mod_attr = ctx.getAttribute("mask_mod");
          auto* prob_mod_attr = ctx.getAttribute("prob_mod");

          // Validate modifier subgraphs in builder as well (avoid OOB / segfault if model is malformed
          // or shape-inference is not executed before function building).
          auto validate_mod_graph = [&](const AttributeProto* attr,
                                        int expected_inputs) -> bool {
            if (attr == nullptr)
              return true;
            if (!attr->has_g())
              return false;
            const auto& g = attr->g();
            if (g.input_size() != expected_inputs)
              return false;
            if (g.output_size() != 1)
              return false;
            return true;
          };

          if (!validate_mod_graph(score_mod_attr, 5))
            return false;
          if (!validate_mod_graph(mask_mod_attr, 4))
            return false;
          if (!validate_mod_graph(prob_mod_attr, 5))
            return false;

          const auto* q_type = ctx.getInputType(0);
          const auto* k_type = ctx.getInputType(1);
          const auto* v_type = ctx.getInputType(2);
          if (!q_type || !k_type || !v_type || !q_type->has_tensor_type() || !k_type->has_tensor_type() ||
              !v_type->has_tensor_type()) {
            return false;
          }

          const auto q_elem_type = q_type->tensor_type().elem_type();
          if (k_type->tensor_type().elem_type() != q_elem_type || v_type->tensor_type().elem_type() != q_elem_type) {
            return false;
          }

          // Scale attribute
          auto scale_attr = ctx.getAttribute("scale");
          float scale = (scale_attr != nullptr) ? scale_attr->f() : static_cast<float>(1);

          // mask_value (float attr) -> cast to softmax_precision for Where
          auto mask_value_attr = ctx.getAttribute("mask_value");
          float mask_value = (mask_value_attr != nullptr) ? mask_value_attr->f() : -3.402823466e+38f;

          // Decide whether score_mod is a trivial identity graph.
          bool score_mod_is_trivial_identity = false;
          if (score_mod_attr != nullptr) {
            const auto& sg = score_mod_attr->g();
            const bool is_empty = (sg.node_size() == 0);
            const bool same_io =
                (sg.input_size() == 5 && sg.output_size() == 1 && sg.input(0).name() == sg.output(0).name());
            score_mod_is_trivial_identity = (is_empty && same_io);
          }

          // Decide whether prob_mod is a trivial identity graph.
          bool prob_mod_is_trivial_identity = false;
          if (prob_mod_attr != nullptr) {
            const auto& pg = prob_mod_attr->g();
            const bool is_empty = (pg.node_size() == 0);
            const bool same_io =
                (pg.input_size() == 5 && pg.output_size() == 1 && pg.input(0).name() == pg.output(0).name());
            prob_mod_is_trivial_identity = (is_empty && same_io);
          }

          // We'll build Loop nodes as local NodeProto and insert them after schema.BuildFunction(),
          // to keep topological order (builder nodes are materialized by BuildFunction at the end).
          bool need_score_loop = (score_mod_attr != nullptr && !score_mod_is_trivial_identity);
          bool need_mask_loop = (mask_mod_attr != nullptr);
          bool need_prob_loop = (prob_mod_attr != nullptr && !prob_mod_is_trivial_identity);

          NodeProto score_loop_node;
          NodeProto mask_loop_node;
          NodeProto prob_loop_node;

          FunctionBuilder builder(functionProto);

          // (These Shape nodes are unused in current fallback; keeping them is OK but optional)
          builder.Add("BatchSize = Shape <start = 0, end = 1> (Q)")
              .Add("QSeqLen = Shape <start = -2, end = -1> (Q)")
              .Add("KVSeqLen = Shape <start = -2, end = -1> (K)");

          // For 4D inputs: Already in desired shape [B,H,seq,head]
          builder.Add("QReshaped = Identity(Q)").Add("KReshaped = Identity(K)").Add("VReshaped = Identity(V)");

          builder.Add("QNumHeads = Shape <start = 1, end = 2> (QReshaped)")
              .Add("KVNumHeads = Shape <start = 1, end = 2> (KReshaped)");

          // If enable_gqa=1, replicate K/V heads to match the query head dimension.
          if (enable_gqa) {
            builder.Const1D("OneI64Vec", static_cast<int64_t>(1))
                .Add("KVRepeat = Div (QNumHeads, KVNumHeads)")
                .Add("Repeats = Concat <axis = 0> (OneI64Vec, KVRepeat, OneI64Vec, OneI64Vec)")
                .Add("KAligned = Tile (KReshaped, Repeats)")
                .Add("VAligned = Tile (VReshaped, Repeats)");
          } else {
            builder.Add("KAligned = Identity(KReshaped)").Add("VAligned = Identity(VReshaped)");
          }

          // Calculate scaling factor if scale attribute not provided
          // Keep scale as 0-D scalar in both paths (explicit scale and calculated default)
          builder
              .Add("QShapeAll = Shape(QReshaped)")
              .Const("Idx3Head", ToTensor<int64_t>(3))
              .Add("QKHeadSize = Gather <axis = 0> (QShapeAll, Idx3Head)")
              .Add("QKHeadSizeF = Cast (QKHeadSize)", "to", float_type)
              .Add("SqrtHeadSize = Sqrt(QKHeadSizeF)")
              .Const1D("NegOne1D", static_cast<int64_t>(-1))
              .Const("OneF", ToTensor<float>(1.0f))
              .Add("CalculatedScale = Div(OneF, SqrtHeadSize)")
              .Const("ScaleF", ToTensor<float>(scale))
              .Add(
                  scale_attr != nullptr ? "ScaleFactorF32 = Identity(ScaleF)"
                                        : "ScaleFactorF32 = Identity(CalculatedScale)");

            builder.Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KAligned)")
              .Add("Score = MatMul(QReshaped, KTranspose)")
              // Apply scale to the attention logits before score_mod/mask_mod/softmax.
              // For PyTorch-like numerics, do the multiplication in float32 (float_type) first.
              .Add("ScoreF = Cast (Score)", "to", float_type)
              .Add("SoftmaxCastF = Mul(ScoreF, ScaleFactorF32)")
              .Add("SoftmaxCast = Cast (SoftmaxCastF)", "to", softmax_precision);

          // Common scalars for scalar-contract Loops (build once if either loop is needed)
          if (need_score_loop || need_mask_loop || need_prob_loop) {
            builder.Add("ScoreShape = Shape(SoftmaxCast)")
                .Const("Idx0", ToTensor<int64_t>(0))
                .Const("Idx1", ToTensor<int64_t>(1))
                .Const("Idx2", ToTensor<int64_t>(2))
                .Const("Idx3", ToTensor<int64_t>(3))
                .Add("B = Gather <axis = 0> (ScoreShape, Idx0)")
                .Add("H = Gather <axis = 0> (ScoreShape, Idx1)")
                .Add("L = Gather <axis = 0> (ScoreShape, Idx2)")
                .Add("S = Gather <axis = 0> (ScoreShape, Idx3)")
                .Add("ScoreFlat = Reshape(SoftmaxCast, NegOne1D)")
                .Add("N = Size(ScoreFlat)")
                .Const("TrueI64", ToTensor<int64_t>(1))
                .Add("CondInit = Cast(TrueI64)", "to", static_cast<int64_t>(TensorProto::BOOL));
          }

          // ----- score_mod (scalar-contract Loop) -----
          if (need_score_loop) {
            const auto& sg = score_mod_attr->g();

            // Build Loop node locally (do NOT add to functionProto yet).
            score_loop_node.Clear();
            score_loop_node.set_op_type("Loop");
            score_loop_node.add_input("N");
            score_loop_node.add_input("CondInit");
            score_loop_node.add_output("ScoreModFlat");

            AttributeProto* body_attr = score_loop_node.add_attribute();
            body_attr->set_name("body");
            body_attr->set_type(AttributeProto::GRAPH);
            GraphProto* body = body_attr->mutable_g();
            body->set_name("FlexAttention_score_mod_body");

            // Body I/O
            body->add_input()->set_name("iter"); // INT64 scalar
            body->add_input()->set_name("cond_in"); // BOOL scalar
            body->add_output()->set_name("cond_out");
            body->add_output()->set_name("scan_out");

            // k = iter % S
            {
              auto* n = body->add_node();
              n->set_op_type("Mod");
              n->add_input("iter");
              n->add_input("S");
              n->add_output("k_idx");
            }
            // t1 = iter / S
            {
              auto* n = body->add_node();
              n->set_op_type("Div");
              n->add_input("iter");
              n->add_input("S");
              n->add_output("t1");
            }
            // q = t1 % L
            {
              auto* n = body->add_node();
              n->set_op_type("Mod");
              n->add_input("t1");
              n->add_input("L");
              n->add_output("q_idx");
            }
            // t2 = t1 / L
            {
              auto* n = body->add_node();
              n->set_op_type("Div");
              n->add_input("t1");
              n->add_input("L");
              n->add_output("t2");
            }
            // head = t2 % H
            {
              auto* n = body->add_node();
              n->set_op_type("Mod");
              n->add_input("t2");
              n->add_input("H");
              n->add_output("head");
            }
            // batch = t2 / H
            {
              auto* n = body->add_node();
              n->set_op_type("Div");
              n->add_input("t2");
              n->add_input("H");
              n->add_output("batch");
            }

            // score_i = Gather(ScoreFlat, iter, axis=0)  -> scalar (0-D)
            {
              auto* n = body->add_node();
              n->set_op_type("Gather");
              n->add_input("ScoreFlat");
              n->add_input("iter");
              n->add_output("score_i");
              auto* a = n->add_attribute();
              a->set_name("axis");
              a->set_type(AttributeProto::INT);
              a->set_i(0);
            }

            // Inline score_mod graph into body (scalar contract)
            {
              std::unordered_map<std::string, std::string> io_map;
              io_map.emplace(sg.input(0).name(), "score_i");
              io_map.emplace(sg.input(1).name(), "batch");
              io_map.emplace(sg.input(2).name(), "head");
              io_map.emplace(sg.input(3).name(), "q_idx");
              io_map.emplace(sg.input(4).name(), "k_idx");
              io_map.emplace(sg.output(0).name(), "score_mod_out");
              InlineGraphInto(body, sg, io_map, "SM_");
            }

            // cond_out = cond_in
            {
              auto* n = body->add_node();
              n->set_op_type("Identity");
              n->add_input("cond_in");
              n->add_output("cond_out");
            }
            // scan_out = score_mod_out
            {
              auto* n = body->add_node();
              n->set_op_type("Identity");
              n->add_input("score_mod_out");
              n->add_output("scan_out");
            }

            // Reshape scan output [N] back to (B,H,L,S).
            builder.Add("ScoreAfterScoreMod = Reshape(ScoreModFlat, ScoreShape)");
          } else {
            builder.Add("ScoreAfterScoreMod = Identity(SoftmaxCast)");
          }

          // ----- mask_mod (scalar-contract Loop) -----
          if (need_mask_loop) {
            const auto& mg = mask_mod_attr->g();

            // Build Loop node locally (do NOT add to functionProto yet).
            mask_loop_node.Clear();
            mask_loop_node.set_op_type("Loop");
            mask_loop_node.add_input("N");
            mask_loop_node.add_input("CondInit");
            mask_loop_node.add_output("MaskFlat");

            AttributeProto* mbody_attr = mask_loop_node.add_attribute();
            mbody_attr->set_name("body");
            mbody_attr->set_type(AttributeProto::GRAPH);
            GraphProto* mbody = mbody_attr->mutable_g();
            mbody->set_name("FlexAttention_mask_mod_body");

            // Body I/O
            mbody->add_input()->set_name("iter");
            mbody->add_input()->set_name("cond_in");
            mbody->add_output()->set_name("cond_out");
            mbody->add_output()->set_name("scan_out");

            // Same index math as score_mod
            {
              auto* n = mbody->add_node();
              n->set_op_type("Mod");
              n->add_input("iter");
              n->add_input("S");
              n->add_output("k_idx");
            }
            {
              auto* n = mbody->add_node();
              n->set_op_type("Div");
              n->add_input("iter");
              n->add_input("S");
              n->add_output("t1");
            }
            {
              auto* n = mbody->add_node();
              n->set_op_type("Mod");
              n->add_input("t1");
              n->add_input("L");
              n->add_output("q_idx");
            }
            {
              auto* n = mbody->add_node();
              n->set_op_type("Div");
              n->add_input("t1");
              n->add_input("L");
              n->add_output("t2");
            }
            {
              auto* n = mbody->add_node();
              n->set_op_type("Mod");
              n->add_input("t2");
              n->add_input("H");
              n->add_output("head");
            }
            {
              auto* n = mbody->add_node();
              n->set_op_type("Div");
              n->add_input("t2");
              n->add_input("H");
              n->add_output("batch");
            }

            // Inline mask_mod graph into body (scalar contract)
            {
              std::unordered_map<std::string, std::string> io_map;
              io_map.emplace(mg.input(0).name(), "batch");
              io_map.emplace(mg.input(1).name(), "head");
              io_map.emplace(mg.input(2).name(), "q_idx");
              io_map.emplace(mg.input(3).name(), "k_idx");
              io_map.emplace(mg.output(0).name(), "mask_mod_out");
              InlineGraphInto(mbody, mg, io_map, "MM_");
            }

            // cond_out = cond_in
            {
              auto* n = mbody->add_node();
              n->set_op_type("Identity");
              n->add_input("cond_in");
              n->add_output("cond_out");
            }
            // scan_out = mask_mod_out
            {
              auto* n = mbody->add_node();
              n->set_op_type("Identity");
              n->add_input("mask_mod_out");
              n->add_output("scan_out");
            }

            // Mask = Reshape(MaskFlat, ScoreShape)
            builder.Add("Mask = Reshape(MaskFlat, ScoreShape)");
            // MaskValueSp = Cast(Const(mask_value), to=softmax_precision)
            builder.Const("MaskValueF", ToTensor<float>(mask_value))
                .Add("MaskValueSp = Cast (MaskValueF)", "to", softmax_precision);
            // ScoreAfterMask = Where(Mask, ScoreAfterScoreMod, MaskValueSp)
            builder.Add("ScoreAfterMask = Where (Mask, ScoreAfterScoreMod, MaskValueSp)");
          } else {
            builder.Add("ScoreAfterMask = Identity(ScoreAfterScoreMod)");
          }

          builder.Add("Prob = Softmax <axis = 3> (ScoreAfterMask)");

          if (need_prob_loop) {
            const auto& pg = prob_mod_attr->g();

            // Flatten probabilities to 1-D for scalar-contract Loop.
            builder.Add("ProbFlat = Reshape(Prob, NegOne1D)");

            prob_loop_node.Clear();
            prob_loop_node.set_op_type("Loop");
            prob_loop_node.add_input("N");
            prob_loop_node.add_input("CondInit");
            prob_loop_node.add_output("ProbModFlat");

            AttributeProto* pbody_attr = prob_loop_node.add_attribute();
            pbody_attr->set_name("body");
            pbody_attr->set_type(AttributeProto::GRAPH);
            GraphProto* pbody = pbody_attr->mutable_g();
            pbody->set_name("FlexAttention_prob_mod_body");

            // Body I/O
            pbody->add_input()->set_name("iter");     // INT64 scalar
            pbody->add_input()->set_name("cond_in");  // BOOL scalar
            pbody->add_output()->set_name("cond_out");
            pbody->add_output()->set_name("scan_out");

            // k = iter % S
            {
              auto* n = pbody->add_node();
              n->set_op_type("Mod");
              n->add_input("iter");
              n->add_input("S");
              n->add_output("k_idx");
            }
            // t1 = iter / S
            {
              auto* n = pbody->add_node();
              n->set_op_type("Div");
              n->add_input("iter");
              n->add_input("S");
              n->add_output("t1");
            }
            // q = t1 % L
            {
              auto* n = pbody->add_node();
              n->set_op_type("Mod");
              n->add_input("t1");
              n->add_input("L");
              n->add_output("q_idx");
            }
            // t2 = t1 / L
            {
              auto* n = pbody->add_node();
              n->set_op_type("Div");
              n->add_input("t1");
              n->add_input("L");
              n->add_output("t2");
            }
            // head = t2 % H
            {
              auto* n = pbody->add_node();
              n->set_op_type("Mod");
              n->add_input("t2");
              n->add_input("H");
              n->add_output("head");
            }
            // batch = t2 / H
            {
              auto* n = pbody->add_node();
              n->set_op_type("Div");
              n->add_input("t2");
              n->add_input("H");
              n->add_output("batch");
            }

            // prob_i = Gather(ProbFlat, iter, axis=0)
            {
              auto* n = pbody->add_node();
              n->set_op_type("Gather");
              n->add_input("ProbFlat");
              n->add_input("iter");
              n->add_output("prob_i");
              auto* a = n->add_attribute();
              a->set_name("axis");
              a->set_type(AttributeProto::INT);
              a->set_i(0);
            }

            // Inline prob_mod graph
            {
              std::unordered_map<std::string, std::string> io_map;
              io_map.emplace(pg.input(0).name(), "prob_i");
              io_map.emplace(pg.input(1).name(), "batch");
              io_map.emplace(pg.input(2).name(), "head");
              io_map.emplace(pg.input(3).name(), "q_idx");
              io_map.emplace(pg.input(4).name(), "k_idx");
              io_map.emplace(pg.output(0).name(), "prob_mod_out");
              InlineGraphInto(pbody, pg, io_map, "PM_");
            }

            // cond_out = cond_in
            {
              auto* n = pbody->add_node();
              n->set_op_type("Identity");
              n->add_input("cond_in");
              n->add_output("cond_out");
            }
            // scan_out = prob_mod_out
            {
              auto* n = pbody->add_node();
              n->set_op_type("Identity");
              n->add_input("prob_mod_out");
              n->add_output("scan_out");
            }
            // Reshape scan output [N] back to (B,H,L,S).
            builder.Add("ProbAfterProbMod = Reshape(ProbModFlat, ScoreShape)");
          } else {
            builder.Add("ProbAfterProbMod = Identity(Prob)");
          }

          if (softmax_precision != T1) {
            builder.Add("VSp = Cast (VAligned)", "to", softmax_precision);
            builder.Add("YSp = MatMul (ProbAfterProbMod, VSp)");
            builder.Add("Y = Cast (YSp)", "to", T1);
          } else {
            builder.Add("Y = MatMul (ProbAfterProbMod, VAligned)");
          }

          schema.BuildFunction(functionProto);

          // Insert score loop after its captured scalars/inputs are defined.
          if (need_score_loop) {
            int last = FindLastProducerIndex(functionProto, {"ScoreFlat", "N", "CondInit", "B", "H", "L", "S"});
            int insert_at = (last >= 0) ? (last + 1) : 0;
            InsertNodeAt(functionProto, score_loop_node, insert_at);
          }

          // Insert mask loop after its captured scalars/inputs are defined.
          // If both loops exist, we recompute after score insertion so ordering stays valid.
          if (need_mask_loop) {
            int last = FindLastProducerIndex(functionProto, {"ScoreFlat", "N", "CondInit", "B", "H", "L", "S"});
            int insert_at = (last >= 0) ? (last + 1) : 0;
            InsertNodeAt(functionProto, mask_loop_node, insert_at);
          }

          // Insert prob loop after its captured scalars/inputs are defined.
          // If other loops exist, we recompute after prior insertions so ordering stays valid.
          if (need_prob_loop) {
            int last = FindLastProducerIndex(functionProto, {"ProbFlat", "N", "CondInit", "B", "H", "L", "S"});
            int insert_at = (last >= 0) ? (last + 1) : 0;
            InsertNodeAt(functionProto, prob_loop_node, insert_at);
          }

          return true;
        }));
} // namespace ONNX_NAMESPACE
