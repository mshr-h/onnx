/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

static constexpr const char* FlexAttention_ver1_doc = R"DOC(
Computes scaled dot-product attention with user-provided customization subgraphs
at up to three stages:
  (1) score_mod: modify each scalar attention score after Q·K^T
  (2) mask_mod : determine which (q_idx, k_idx) connections are allowed
  (3) prob_mod : modify each scalar probability after Softmax
This mirrors PyTorch's torch.nn.attention.flex_attention behavior.  (PyTorch is
prototype; ONNX uses preview domain for incubation.)

Inputs
------
1) query (Q) : T
   Query tensor. Shape: (B, Hq, L, E)

2) key (K) : T
   Key tensor. Shape: (B, Hkv, S, E)

3) value (V) : T
   Value tensor. Shape: (B, Hkv, S, Ev)

Outputs
-------
1) output (Y) : T
   Attention output. Shape: (B, Hq, L, Ev)

Attributes
----------
scale : FLOAT (optional)
  Scaling factor applied to raw dot-product scores prior to score_mod/mask/softmax.
  If omitted, the default is 1/sqrt(head_size), matching ONNX Attention / SDPA conventions.
  For numerical stability, implementations may scale Q and K by sqrt(scale) before MatMul
  (so that the score matrix is scaled by scale overall).

enable_gqa : INT (default = 0)
  If 0: requires Hq == Hkv.
  If 1: enables Grouped Query Attention; key/value heads are broadcast to query heads.
         Requires Hq % Hkv == 0.

mask_value : FLOAT (default = -3.402823466e+38)
  Value used to replace masked scores before Softmax (i.e., an approximation of -inf).
  Backends may clamp as needed for the chosen T.

score_mod : GRAPH (optional)
  A subgraph that modifies each scalar attention score.
  Signature (positional inputs):
    (score, batch, head, q_idx, k_idx) -> (score_out)
  Input/Output requirements:
    - score      : TScalar (a scalar tensor with element type compatible with T)
    - batch/head/q_idx/k_idx : TI (scalar int indices)
    - score_out  : same type as score
  The graph MUST have exactly 1 output.
  Intended to match PyTorch score_mod signature:
    def score_mod(score, batch, head, q_idx, k_idx) -> score'    (PyTorch uses torch.int for indices).

mask_mod : GRAPH (optional)
  A subgraph that decides whether a score position is allowed.
  Signature (positional inputs):
    (batch, head, q_idx, k_idx) -> (mask_out)
  Requirements:
    - batch/head/q_idx/k_idx : TI (scalar int indices)
    - mask_out : tensor(bool) scalar; True = allowed, False = masked.
  The graph MUST have exactly 1 output.
  Intended to match PyTorch mask_mod signature used by create_block_mask:
    def mask_mod(b, h, q_idx, k_idx) -> bool

prob_mod : GRAPH (optional)
  A subgraph that modifies each scalar probability AFTER Softmax.
  Signature (positional inputs):
    (prob, batch, head, q_idx, k_idx) -> (prob_out)
  Requirements:
    - prob      : TScalar
    - indices   : TI
    - prob_out  : same type as prob
  The graph MUST have exactly 1 output.
  Note: prob_out is NOT automatically renormalized by this operator; if a model
  requires renormalization, prob_mod should implement it.

Type Constraints
----------------
T : tensor(float), tensor(float16), tensor(bfloat16),
    tensor(float8e4m3fn), tensor(float8e4m3fnuz),
    tensor(float8e5m2), tensor(float8e5m2fnuz),
    tensor(float8e8m0)
  (ONNX has defined float8 formats; see ONNX float8 documentation and operator schema usage.)

TScalar :
  scalar tensor whose element type is the same as T's element type.
  (i.e., 0-dim tensor(T_elem))

TI : tensor(int64) or tensor(int32)
  Scalar index tensors.

TMod :
  tensor(float), tensor(float16), tensor(bfloat16),
  tensor(float8e4m3fn), tensor(float8e4m3fnuz),
  tensor(float8e5m2), tensor(float8e5m2fnuz),
  tensor(float8e8m0),
  tensor(int64), tensor(int32), tensor(bool)
  (Deliberately broad: mod graphs may consume parameters of different dtypes.)

Shape Constraints & Semantics
-----------------------------
Let:
  Q: (B, Hq, L, E)
  K: (B, Hkv, S, E)
  V: (B, Hkv, S, Ev)

Head mapping:
  if enable_gqa == 0: kvh = hq  (requires Hq==Hkv)
  if enable_gqa == 1:
     group = Hq / Hkv
     kvh = floor(hq / group)     (equivalent to repeating K,V along head axis)

For each b in [0,B), hq in [0,Hq), q in [0,L), kv in [0,S):
  raw = dot(Q[b,hq,q,:], K[b,kvh,kv,:])
  s   = raw * (scale if provided else 1/sqrt(E))
  if score_mod provided:
      s = score_mod(s, b, hq, q, kv)
  if mask_mod provided:
      m = mask_mod(b, hq, q, kv)   # bool
      if m == False: s = mask_value

Let P[b,hq,q,:] = Softmax over kv dimension of S[b,hq,q,:].

If prob_mod provided:
  For each kv:
     P[b,hq,q,kv] = prob_mod(P[b,hq,q,kv], b, hq, q, kv)

Output:
  Y[b,hq,q,:] = Σ_{kv} P[b,hq,q,kv] * V[b,kvh,kv,:]

Notes:
  - If score_mod/mask_mod/prob_mod are absent, they behave as identity / allow-all / identity.
  - Graph attributes must be side-effect-free and deterministic for portability.
  - Reference ONNX Function fallback may be slow (elementwise modifications); optimized EPs
    can fuse the pattern.

Documentation (operator docstring)
----------------------------------
"FlexAttention computes scaled dot-product attention between query, key, and value,
with optional customization subgraphs applied per-element to scores (score_mod),
masking decisions (mask_mod), and probabilities after softmax (prob_mod). This enables
export of flexible attention patterns (e.g., ALiBi, custom masking, post-softmax quantization)
without decomposing into many small ops, while still allowing backends to fuse the full pattern."
)DOC";

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
        "Attribute ", attr_name, " expected ", expected_inputs, " inputs but graph has ", g.input_size(), ".");
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

  const auto& q_shape = q_type->tensor_type().shape();
  const auto& k_shape = k_type->tensor_type().shape();
  const auto& v_shape = v_type->tensor_type().shape();
  if (q_shape.dim_size() != 4 || k_shape.dim_size() != 4 || v_shape.dim_size() != 4) {
    fail_shape_inference("FlexAttention requires rank-4 inputs with shape (B, H, L, D).");
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

  // Validate scale attribute.
  if (const auto* scale_attr = ctx.getAttribute("scale")) {
    if (scale_attr->has_f() && !(scale_attr->f() > 0.0f)) {
      fail_shape_inference("scale must be > 0 when provided.");
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

  ValidateFlexAttentionGraph(ctx, ctx.getAttribute("score_mod"), 5, "score_mod", q_elem_type, true);
  ValidateFlexAttentionGraph(ctx, ctx.getAttribute("mask_mod"), 4, "mask_mod", TensorProto::BOOL, true);
  ValidateFlexAttentionGraph(ctx, ctx.getAttribute("prob_mod"), 5, "prob_mod", q_elem_type, true);
}

ONNX_PREVIEW_OPERATOR_SET_SCHEMA(
    FlexAttention,
    1,
    OpSchema()
        .SetDoc(FlexAttention_ver1_doc)
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
            "T1")
        .Output(
            0,
            "Y",
            "The output tensor . "
            "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, hidden_size)`. "
            "For cases with a 3D input tensor, `hidden_size = q_num_heads * v_head_size`",
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
            "If softmax precision is not provided, the same precision as the input of softmax (Q and K) is used.",
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
            "The graph MUST have exactly 5 inputs in this exact order and names:\n"
            "  (score, batch, head, q_idx, k_idx)\n"
            "and exactly 1 output (score_out).\n"
            "All five inputs are scalar tensors; the output is a scalar tensor.\n"
            "In the schema-defined function fallback, these scalars are tensorized to shape (B,H,L,S) "
            "and the graph is inlined using only existing ONNX operators.",
            AttributeProto::GRAPH,
            OPTIONAL_VALUE)
        .Attr("mask_mod", "Optional mask modifier graph.", AttributeProto::GRAPH, OPTIONAL_VALUE)
        .Attr("prob_mod", "Optional probability modifier graph.", AttributeProto::GRAPH, OPTIONAL_VALUE)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain Q and K inputs types to float tensors.")
        .TypeAndShapeInferenceFunction(FlexAttentionShapeInference)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          int64_t int_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
          int64_t float_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

          // Get input types
          auto t_qk = ctx.getInputType(0);
          if ((t_qk == nullptr) || (!t_qk->has_tensor_type()))
            return false;
          int64_t T1 = t_qk->tensor_type().elem_type();

          // Determine precision types for Softmax
          auto softmax_precision_attr = ctx.getAttribute("softmax_precision");
          int64_t softmax_precision = (softmax_precision_attr != nullptr) ? softmax_precision_attr->i() : T1;
          if ((softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE)) {
            return false; // Error
          }

          // gqa not supported in this builder
          if (ctx.getAttribute("enable_gqa") != nullptr && ctx.getAttribute("enable_gqa")->i() != 0) {
            return false;
          }

          auto* score_mod_attr = ctx.getAttribute("score_mod");
          auto* mask_mod_attr = ctx.getAttribute("mask_mod");
          auto* prob_mod_attr = ctx.getAttribute("prob_mod");

          // Only support score_mod for now
          if (mask_mod_attr != nullptr || prob_mod_attr != nullptr) {
            return false;
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

          FunctionBuilder builder(functionProto);
          builder
              .Add("BatchSize = Shape <start = 0, end = 1> (Q)") // batch size
              .Add("QSeqLen = Shape <start = -2, end = -1> (Q)") // q_sequence_length
              .Add("KVSeqLen = Shape <start = -2, end = -1> (K)"); // kv_sequence_length

          // For 4D inputs: Already in desired shape [batch_size, num_heads, seq_length, head_size]
          builder.Add("QReshaped = Identity(Q)").Add("KReshaped = Identity(K)").Add("VReshaped = Identity(V)");

          builder
              .Add("QNumHeads = Shape <start = 1, end = 2> (QReshaped)") // q_num_heads
              .Add("KVNumHeads = Shape <start = 1, end = 2> (KReshaped)"); // kv_num_heads

          // Calculate scaling factor if scale attribute not provided
          auto scale_attr = ctx.getAttribute("scale");
          float scale = (scale_attr != nullptr) ? scale_attr->f() : static_cast<float>(1);
          builder
              .Add("QKHeadSize = Shape <start = 3, end = 4> (QReshaped)") // head_size for Q and K
              .Add("QKHeadSizeF = Cast (QKHeadSize)", "to", float_type)
              .Add("VHeadSize = Shape <start = 3, end = 4> (VReshaped)") // head_size for V
              .Add("SqrtHeadSize = Sqrt(QKHeadSizeF)")
              .Const1D("One1D", static_cast<int64_t>(1))
              .Const1D("NegOne1D", static_cast<int64_t>(-1))
              .Const1D("One1DF", static_cast<float>(1))
              .Const1D("Zero1D", static_cast<int64_t>(0))
              .Add("CalculatedScale = Div(One1DF, SqrtHeadSize)")
              .Const("ScaleF", ToTensor<float>(scale))
              .Add(scale_attr != nullptr ? "ScaleFactor = Identity(ScaleF)" : "ScaleFactor = Identity(CalculatedScale)")
              .Add("ScaleFactorSqrt = Sqrt(ScaleFactor)")
              .Add("ScaleFactorF = Cast (ScaleFactorSqrt)", "to", T1);

          // TODO: Add score_mod, mask_mod, prob_mod support here
          // The following pattern is applied
          //      Q          K          V
          //      |          |          |
          //     Q*scale    K*scale     |
          //      |          |          |
          //      |       Transpose     |
          //      |          |          |
          //      ---MatMul---          |
          //            |               |
          //            |               |
          //         Softmax            |
          //            |               |
          //            -----MatMul------
          //                    |
          //                    Y
          builder.Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KReshaped)")
              .Add("QScaled = Mul(QReshaped, ScaleFactorF)")
              .Add("KScaled = Mul(KTranspose, ScaleFactorF)")
              .Add("Score = MatMul(QScaled, KScaled)")
              .Add("ScoreCast = Cast (Score)", "to", T1)
              .Add("SoftmaxCast = Cast (Score)", "to", softmax_precision);

          // Apply score_mod if provided
          if (score_mod_attr != nullptr) {
            const auto& g = score_mod_attr->g();

            // ★ Empty graph (i.e., identity) is not inlined; skip to Identity case below
            const bool is_empty = (g.node_size() == 0);
            const bool same_io =
                (g.input_size() >= 1 && g.output_size() >= 1 && g.input(0).name() == g.output(0).name());
            if (is_empty && same_io) {
              builder.Add("ScoreAfterMod = Identity(SoftmaxCast)");
            } else {
              // ScoreShape is (B,H,L,S)
              builder.Add("B = Shape <start = 0, end = 1> (SoftmaxCast)")
                  .Add("H = Shape <start = 1, end = 2> (SoftmaxCast)")
                  .Add("L = Shape <start = 2, end = 3> (SoftmaxCast)")
                  .Add("S = Shape <start = 3, end = 4> (SoftmaxCast)")
                  .Add("Target = Concat <axis = 0> (B, H, L, S)");

              // b/h/q/k index grids (int64)
              builder.Add("b = Range (Zero1D, B, One1D)")
                  .Add("ShapeB111 = Concat <axis = 0> (B, One1D, One1D, One1D)")
                  .Add("b4 = Reshape (b, ShapeB111)")
                  .Add("BatchIdx = Expand (b4, Target)")
                  .Add("h = Range (Zero1D, H, One1D)")
                  .Add("Shape1H11 = Concat <axis = 0> (One1D, H, One1D, One1D)")
                  .Add("h4 = Reshape (h, Shape1H11)")
                  .Add("HeadIdx = Expand (h4, Target)")
                  .Add("q = Range (Zero1D, L, One1D)")
                  .Add("Shape11L1 = Concat <axis = 0> (One1D, One1D, L, One1D)")
                  .Add("q4 = Reshape (q, Shape11L1)")
                  .Add("QIdx = Expand (q4, Target)")
                  .Add("k = Range (Zero1D, S, One1D)")
                  .Add("Shape111S = Concat <axis = 0> (One1D, One1D, One1D, S)")
                  .Add("k4 = Reshape (k, Shape111S)")
                  .Add("KIdx = Expand (k4, Target)");

              builder.AddInlinedCall(
                  {"ScoreAfterMod"},
                  score_mod_attr->g(),
                  {"SoftmaxCast", "BatchIdx", "HeadIdx", "QIdx", "KIdx"},
                  "SM_");
            }
          } else {
            builder.Add("ScoreAfterMod = Identity(SoftmaxCast)");
          }

          builder.Add("Prob = Softmax <axis = 3> (ScoreAfterMod)");
          if (softmax_precision != T1) {
            builder.Add("VSp = Cast (VReshaped)", "to", softmax_precision);
            builder.Add("YSp = MatMul (Prob, VSp)");
            builder.Add("Y = Cast (YSp)", "to", T1);
          } else {
            builder.Add("Y = MatMul (Prob, VReshaped)");
          }

          schema.BuildFunction(functionProto);
          return true;
        }));
} // namespace ONNX_NAMESPACE
