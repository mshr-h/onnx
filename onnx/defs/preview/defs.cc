/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

static void ValidateFlexAttentionGraph(
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
        "Attribute ",
        attr_name,
        " expected ",
        expected_inputs,
        " inputs but graph has ",
        g.input_size(),
        ".");
  }

  if (g.output_size() != 1) {
    fail_shape_inference("Attribute ", attr_name, " must have exactly one output.");
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

  if (!hasNInputShapes(ctx, 3)) {
    return;
  }

  const auto& q_shape = q_type->tensor_type().shape();
  const auto& k_shape = k_type->tensor_type().shape();
  const auto& v_shape = v_type->tensor_type().shape();
  const auto q_rank = q_shape.dim_size();
  const auto k_rank = k_shape.dim_size();
  const auto v_rank = v_shape.dim_size();
  if ((q_rank != 3 && q_rank != 4) || q_rank != k_rank || q_rank != v_rank) {
    fail_shape_inference("FlexAttention requires all inputs to have rank 3 or rank 4 (matched).");
  }

  const bool is_3d = q_rank == 3;

  // Validate head sizes between K and V (only for 4D path).
  if (!is_3d) {
    if (k_shape.dim(1).has_dim_value() && v_shape.dim(1).has_dim_value() &&
        k_shape.dim(1).dim_value() != v_shape.dim(1).dim_value()) {
      fail_shape_inference("Key and value must share the same head dimension.");
    }
  }

  // Validate sequence length alignment between K and V.
  const int seq_dim = is_3d ? 1 : 2;
  if (k_shape.dim(seq_dim).has_dim_value() && v_shape.dim(seq_dim).has_dim_value() &&
      k_shape.dim(seq_dim).dim_value() != v_shape.dim(seq_dim).dim_value()) {
    fail_shape_inference("Key and value must share the same sequence length.");
  }

  // Validate depth alignment between Q and K.
  const int depth_dim = is_3d ? 2 : 3;
  if (q_shape.dim(depth_dim).has_dim_value() && k_shape.dim(depth_dim).has_dim_value() &&
      q_shape.dim(depth_dim).dim_value() != k_shape.dim(depth_dim).dim_value()) {
    fail_shape_inference("Query and key must share the same embedding dimension.");
  }

  const auto enable_gqa = getAttribute(ctx, "enable_gqa", 0);
  if (!is_3d) {
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
  }

  auto* output_shape = output_type->mutable_shape();
  output_shape->clear_dim();
  if (is_3d) {
    *output_shape->add_dim() = q_shape.dim(0);
    *output_shape->add_dim() = q_shape.dim(1);
    *output_shape->add_dim() = v_shape.dim(2);
    mergeInDimensionInfo(k_shape.dim(0), *output_shape->mutable_dim(0), 0);
    mergeInDimensionInfo(v_shape.dim(0), *output_shape->mutable_dim(0), 0);
  } else {
    *output_shape->add_dim() = q_shape.dim(0);
    *output_shape->add_dim() = q_shape.dim(1);
    *output_shape->add_dim() = q_shape.dim(2);
    *output_shape->add_dim() = v_shape.dim(3);

    mergeInDimensionInfo(k_shape.dim(0), *output_shape->mutable_dim(0), 0);
    mergeInDimensionInfo(v_shape.dim(0), *output_shape->mutable_dim(0), 0);
    if (enable_gqa == 0) {
      mergeInDimensionInfo(k_shape.dim(1), *output_shape->mutable_dim(1), 1);
    }
  }

  const size_t mod_input_count = (ctx.getNumInputs() > 3) ? (ctx.getNumInputs() - 3) : 0;
  ValidateFlexAttentionGraph(ctx, ctx.getAttribute("score_mod"), 5 + mod_input_count, "score_mod", q_elem_type, true);
  ValidateFlexAttentionGraph(ctx, ctx.getAttribute("mask_mod"), 4 + mod_input_count, "mask_mod", TensorProto::BOOL, true);
  ValidateFlexAttentionGraph(ctx, ctx.getAttribute("prob_mod"), 5 + mod_input_count, "prob_mod", q_elem_type, true);
}

static bool BuildFlexAttentionFunction(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  // Provide a dense, no-modifier reference path only when graph attributes and extra inputs are absent.
  if (ctx.getAttribute("score_mod") != nullptr || ctx.getAttribute("mask_mod") != nullptr ||
      ctx.getAttribute("prob_mod") != nullptr) {
    return false;
  }
  if (ctx.getAttribute("scale") != nullptr) {
    return false; // scale handling would require dtype-aware constants
  }
  if (ctx.getAttribute("enable_gqa") != nullptr && ctx.getAttribute("enable_gqa")->i() != 0) {
    return false;
  }
  if (ctx.hasInput(3)) {
    return false; // mod_inputs present; not handled by reference function
  }

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

  const auto& q_shape = q_type->tensor_type().shape();
  const auto& k_shape = k_type->tensor_type().shape();
  const auto& v_shape = v_type->tensor_type().shape();
  if (q_shape.dim_size() == 0 || k_shape.dim_size() == 0 || v_shape.dim_size() == 0) {
    return false;
  }
  const bool is_3d = (q_shape.dim_size() == 3 && k_shape.dim_size() == 3 && v_shape.dim_size() == 3);
  const bool is_4d = (q_shape.dim_size() == 4 && k_shape.dim_size() == 4 && v_shape.dim_size() == 4);
  if (!is_3d && !is_4d) {
    return false;
  }

  FunctionBuilder builder(functionProto);
  if (is_3d) {
    builder.Add("Batch = Shape <start = 0, end = 1> (Q)")
        .Add("QSeq = Shape <start = 1, end = 2> (Q)")
        .Add("KSeq = Shape <start = 1, end = 2> (K)")
        .Add("Embed = Shape <start = 2, end = 3> (Q)")
        .Const1D("One", static_cast<int64_t>(1))
        .Add("QReshapeShape = Concat <axis = 0> (Batch, One, QSeq, Embed)")
        .Add("KReshapeShape = Concat <axis = 0> (Batch, One, KSeq, Embed)")
        .Add("QReshaped = Reshape (Q, QReshapeShape)")
        .Add("KReshaped = Reshape (K, KReshapeShape)")
        .Add("VReshapeShape = Concat <axis = 0> (Batch, One, KSeq, Shape <start = 2, end = 3> (V))")
        .Add("VReshaped = Reshape (V, VReshapeShape)")
        .Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KReshaped)")
        .Add("Score = MatMul (QReshaped, KTranspose)")
        .Add("Prob = Softmax <axis = 3> (Score)")
        .Add("Y4D = MatMul (Prob, VReshaped)")
        .Add("Y = Squeeze <axes = [1]> (Y4D)");
  } else {
    builder.Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (K)")
        .Add("Score = MatMul (Q, KTranspose)")
        .Add("Prob = Softmax <axis = 3> (Score)")
        .Add("Y = MatMul (Prob, V)");
  }
  schema.BuildFunction(functionProto);
  return true;
}

static constexpr const char* FlexAttention_ver1_doc = R"DOC(
computes scaled dot-product attention, with optional custom
subgraphs injected at three stages:

  1) score_mod: applied to raw attention scores (QK^T * scale, before any mask)
  2) mask_mod : applied after built-in masking (attn_mask and/or is_causal), before softmax
  3) prob_mod : applied to probabilities after softmax, before multiplying by V

Each *_mod attribute is a GraphProto (graph attribute). If an attribute is not
provided, the corresponding stage is treated as identity.
)DOC";

ONNX_PREVIEW_OPERATOR_SET_SCHEMA(
    FlexAttention,
    1,
    OpSchema()
        .SetDoc(FlexAttention_ver1_doc)
        .Attr(
            "scale",
            "Scaling factor applied to $Q*K^T$. Default value is `1/sqrt(head_size)`. To prevent "
            "[numerical overflow](https://tinyurl.com/sudb9s96), scale `Q`, `K` by `sqrt(scale)` before matmul.",
            AttributeProto::FLOAT,
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
            "q_num_heads",
            "Number of heads of query. Must be used with 3D inputs of Q, K and V. ",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "kv_num_heads",
            "Number of heads of key and value. Must be used with 3D inputs of Q, K and V. ",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr(
            "softmax_precision",
            "The floating-point precision used in softmax computation. "
            "If softmax precision is not provided, the same precision as the input of softmax (Q and K) is used.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Attr("score_mod",
          R"DOC(
GraphProto applied to attention scores right after QK^T*scale,
before any masking. Expected signature:

  Inputs : (scores)
  Outputs: (scores_out)

Where scores is a 4D tensor shaped (batch_size, q_num_heads, q_seq_len, total_seq_len)
in the internal canonical form.
The subgraph must return the same shape and element type as input.
)DOC",
            AttributeProto::GRAPH, OPTIONAL)
        .Attr("mask_mod",
          R"DOC(
GraphProto applied after built-in masking (attn_mask and/or is_causal),
before softmax. Expected signature:

  Inputs : (scores)
  Outputs: (scores_out)

Must preserve shape and element type.
Note: This stage is intended for additional or alternative masking logic;
implementations may choose to fuse this with mask application.
)DOC",
            AttributeProto::GRAPH, OPTIONAL)
        .Attr("prob_mod",
          R"DOC(
GraphProto applied to probabilities after softmax, before MatMul with V.
Expected signature:

  Inputs : (probs)
  Outputs: (probs_out)

Where probs is a 4D tensor shaped (batch_size, q_num_heads, q_seq_len, total_seq_len)
in the internal canonical form.
Must preserve shape and element type.
)DOC",
            AttributeProto::GRAPH, OPTIONAL)
        .Input(
            0,
            "Q",
            "Query tensor. "
            "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, q_hidden_size)`. "
            "For cases with a 3D input tensor, `q_hidden_size = q_num_heads * head_size`",
            "T1")
        .Input(
            1,
            "K",
            "Key tensor. "
            "4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, head_size)` or 3D tensor with shape `(batch_size, kv_sequence_length, k_hidden_size)`. "
            "For cases with a 3D input tensor, `k_hidden_size = kv_num_heads * head_size`",
            "T1")
        .Input(
            2,
            "V",
            "Value tensor. "
            "4D tensor with shape `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, kv_sequence_length, v_hidden_size)`. "
            "For cases with a 3D input tensor, `v_hidden_size = kv_num_heads * v_head_size`",
            "T2")
        .Input(3, "past_key",
              "Optional KV-cache past key: (B, kv_num_heads, past_seq_len, head_size).",
              "T1", OpSchema::Optional)
        .Input(4, "past_value",
              "Optional KV-cache past value: (B, kv_num_heads, past_seq_len, v_head_size).",
              "T2", OpSchema::Optional)
        .Input(5, "nonpad_kv_seqlen",
              "Optional vector (B,) indicating valid tokens per sample; "
              "should not be used together with past/present KV-cache.",
              "tensor(int64)", OpSchema::Optional)
        .Input(
            3,
            "mod_inputs",
            "Extra tensors forwarded to modifier graphs. Ignored when corresponding graphs are absent.",
            "TMod",
            OpSchema::Variadic,
            false,
            0)
        .Output(0, "output", "Attention output. Shape (B, Hq, L, Ev).", "T")
        .TypeConstraint(
            "T1",
            {"tensor(float)",
             "tensor(float16)",
             "tensor(bfloat16)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(float8e8m0)"},
            "Constrain Q/K and output to floating-point tensors.")
        .TypeConstraint(
            "T2",
            {"tensor(float)",
             "tensor(float16)",
             "tensor(bfloat16)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(float8e8m0)"},
            "Constrain V to floating-point tensors.")
        .TypeConstraint(
            "TMod",
            {"tensor(float)",
             "tensor(float16)",
             "tensor(bfloat16)",
             "tensor(float8e4m3fn)",
             "tensor(float8e4m3fnuz)",
             "tensor(float8e5m2)",
             "tensor(float8e5m2fnuz)",
             "tensor(float8e8m0)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(bool)"},
            "Constrain modifier input tensors to common numeric/bool types.")
        .TypeAndShapeInferenceFunction(FlexAttentionShapeInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder(BuildFlexAttentionFunction));

} // namespace ONNX_NAMESPACE
