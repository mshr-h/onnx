/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <optional>

#include "onnx/defs/function.h"
#include "onnx/defs/nn/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

static constexpr const char* FlexAttention_ver1_doc = R"DOC(

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

For self attention, `kv_sequence_length` equals to `q_sequence_length`.

For cross attention, query and key might have different lengths.

This operator also covers the 3 following variants based on the number of heads:
1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

Attention bias to be added is calculated based on `attn_mask` input and `is_causal` attribute:
1) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.
2) If `is_causal` is set to `1`, attention scores above the diagonal are masked out, regardless of the `attn_mask` input.

With respect to KV cache update, this operator allows the following two use cases:

1) Cache update happens inside the Attention operator. In this case, the `K` and `V` inputs contain only the incoming
tokens for the current autoregressive step, and the four optional inputs/outputs past and present key and value are
all needed. The Attention op performs a Concat operation on the past and incoming key and value to form the present
key and value, respectively. Note that this only works correctly for the special case where the past key and value
do not contain padded tokens.
2) Cache update happens outside the Attention operator (for example, through the `TensorScatter` operator). In this
case, the `K` and `V` inputs correspond to the entire cache tensor, so the four optional inputs/outputs past and
present key and value should not be used. An additional input `nonpad_kv_seqlen` of shape (batch_size,) may be
provided to indicate the number of non-padding tokens in each sample of the batch to save unnecessary computation.
Here, the kv_sequence dimension of `attn_mask` can be shorter than `K` and `V`, but still needs to be at least as long
as the maximum value of `nonpad_kv_seqlen`.

Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:

```
  The following pattern is applied by this operator:
      Q          K          V
      |          |          |
Q*sqrt(scale) K*sqrt(scale) |
      |          |          |
      |       Transpose     |
      |          |          |
      ---MatMul---          |
            |               |
 at_mask---Add              |
            |               |
  softcap (if provided)     |
            |               |
         Softmax            |
            |               |
            -----MatMul------
                   |
                   Y
```

)DOC";

ONNX_PREVIEW_OPERATOR_SET_SCHEMA(
    FlexAttention,
    1,
    OpSchema()
        .SetDoc(FlexAttention_ver1_doc)
        .Attr(
            "is_causal",
            "If set to `1`, the attention masking is a lower triangular matrix when the mask is a square matrix. "
            "The attention masking has the form of the upper left causal bias due to the alignment.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "scale",
            "Scaling factor applied to $Q*K^T$. Default value is `1/sqrt(head_size)`. To prevent "
            "[numerical overflow](https://tinyurl.com/sudb9s96), scale `Q`, `K` by `sqrt(scale)` before matmul.",
            AttributeProto::FLOAT,
            OPTIONAL_VALUE)
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
        .Attr(
            "softcap",
            "Softcap value for attention weights. Default value is 0.",
            AttributeProto::FLOAT,
            static_cast<float>(0))
        .Attr(
            "qk_matmul_output_mode",
            "If set to `0`, qk_matmul_output is the output of qk matmul. "
            "If set to `1`, qk_matmul_output includes the addition of the attention mask to the output of qk matmul. "
            "If set to `2`, qk_matmul_output is the output after the softcap operation. "
            "If set to `3`, qk_matmul_output is the output after the softmax operation. "
            "Default value is 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
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
        .Input(
            3,
            "attn_mask",
            "Attention mask. "
            "Shape must be broadcastable to `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` "
            "where `total_sequence_length = past_sequence_length + kv_sequence_length.` "
            "The last dimension can also be shorter than `total_sequence_length` and will be padded to `total_sequence_length` with negative infinity. "
            "Two types of masks are supported: a boolean mask where a value of `True` indicates that the element should take part in attention, "
            "or a float mask of the same type as query, key, value that is added to the attention score.",
            "U",
            OpSchema::Optional)
        .Input(
            4,
            "past_key",
            "past state cache for key with shape `(batch_size, kv_num_heads, past_sequence_length, head_size)`",
            "T1",
            OpSchema::Optional)
        .Input(
            5,
            "past_value",
            "past state cache for value with shape `(batch_size, kv_num_heads, past_sequence_length, v_head_size)`",
            "T2",
            OpSchema::Optional)
        .Input(
            6,
            "nonpad_kv_seqlen",
            "A vector of integers of shape `(batch_size,)` that indicates the number of valid (ie, non-padding) "
            "tokens in each sample. A padding mask can be derived from this. This should not be used together with "
            "`past_key` and `past_value` inputs or `present_key` and `present_value` outputs "
            "(See the KV cache use cases in the operator description).",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            0,
            "Y",
            "The output tensor . "
            "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, v_head_size)` or 3D tensor with shape `(batch_size, q_sequence_length, hidden_size)`. "
            "For cases with a 3D input tensor, `hidden_size = q_num_heads * v_head_size`",
            "T1")
        .Output(
            1,
            "present_key",
            "Updated key cache with shape `(batch_size, kv_num_heads, total_sequence_length, head_size)` "
            "where `total_sequence_length = past_sequence_length + kv_sequence_length`.",
            "T1",
            OpSchema::Optional)
        .Output(
            2,
            "present_value",
            "Updated value cache with shape `(batch_size, kv_num_heads, total_sequence_length, v_head_size)` "
            "where `total_sequence_length = past_sequence_length + kv_sequence_length`.",
            "T2",
            OpSchema::Optional)
        .Output(
            3,
            "qk_matmul_output",
            "The output of QK matmul. "
            "4D tensor with shape `(batch_size, q_num_heads, q_sequence_length, total_sequence_length)` "
            "where `total_sequence_length = past_sequence_length + kv_sequence_length`.",
            "T1",
            OpSchema::Optional)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain Q and K inputs types to float tensors.")
        .TypeConstraint("T2", OpSchema::all_float_types_ir4(), "Constrain V input types to float tensors.")
        .TypeConstraint(
            "U",
            OpSchema::all_non_complex_numeric_types_plus_bool_ir4(),
            "Constrain output 'mask' types to boolean tensors and input types.")
        .TypeAndShapeInferenceFunction(defs::nn::utils::AttentionPropagateElemTypeFromInputToOutput)
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          // ScaledDotProductAttention <scale, is_causal, q_num_heads, kv_numheads> (Q, K, V, attn_mask, past_key,
          // past_value) => (Y, present_key?, present_value?)
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
              (softmax_precision != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))
            return false; // Error

          // If shape is 3D, q_num_heads and kv_num_heads is provided,
          // for 4D cases, set num_heads to zero for reshape purposes
          auto q_num_heads_attr = ctx.getAttribute("q_num_heads");
          int64_t q_num_heads = (q_num_heads_attr != nullptr) ? q_num_heads_attr->i() : 0;
          auto kv_num_heads_attr = ctx.getAttribute("kv_num_heads");
          int64_t kv_num_heads = (kv_num_heads_attr != nullptr) ? kv_num_heads_attr->i() : 0;

          // Determine if input is 3D (requires reshape and transpose) or 4D (direct reshape)
          bool is_3d_input = (q_num_heads > 0 && kv_num_heads > 0);

          FunctionBuilder builder(functionProto);
          builder
              .Add("BatchSize = Shape <start = 0, end = 1> (Q)") // batch size
              .Add("QSeqLen = Shape <start = -2, end = -1> (Q)") // q_sequence_length
              .Add("KVSeqLen = Shape <start = -2, end = -1> (K)"); // kv_sequence_length

          if (is_3d_input) {
            // For 3D inputs: First reshape to [batch_size, seq_length, num_heads, head_size]
            // then transpose to [batch_size, num_heads, seq_length, head_size]
            builder
                .Const1D("QNumHeadsAttr", q_num_heads) // q_num_heads from attrs
                .Const1D("KVNumHeadsAttr", kv_num_heads) // kv_num_heads from attrs
                .Const1D("NegOne", static_cast<int64_t>(-1)); // head_size, inferred from other dimensions

            builder.Add("QIntermediateShape = Concat <axis = 0> (BatchSize, QSeqLen, QNumHeadsAttr, NegOne)")
                .Add("KVIntermediateShape = Concat <axis = 0> (BatchSize, KVSeqLen, KVNumHeadsAttr, NegOne)")
                .Add("QIntermediate = Reshape (Q, QIntermediateShape)")
                .Add("KIntermediate = Reshape (K, KVIntermediateShape)")
                .Add("VIntermediate = Reshape (V, KVIntermediateShape)")
                // Then transpose to [batch_size, num_heads, seq_length, head_size]
                .Add("QReshaped = Transpose <perm = [0, 2, 1, 3]> (QIntermediate)")
                .Add("KReshaped = Transpose <perm = [0, 2, 1, 3]> (KIntermediate)")
                .Add("VReshaped = Transpose <perm = [0, 2, 1, 3]> (VIntermediate)");
          } else {
            // For 4D inputs: Already in desired shape [batch_size, num_heads, seq_length, head_size]
            builder.Add("QReshaped = Identity(Q)").Add("KReshaped = Identity(K)").Add("VReshaped = Identity(V)");
          }

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

          // Update key and value caches for past and present states

          if (ctx.hasInput(4)) {
            builder.Add("PresentKey = Concat <axis = 2> (past_key, KReshaped)");
            builder.Add("PastKVSeqLen =  Shape <start = -2, end = -1> (past_key)");
          } else {
            builder.Add("PresentKey = Identity (KReshaped)");
            builder.Const1D("PastKVSeqLen", static_cast<int64_t>(0));
          }
          if (ctx.hasOutput(1)) {
            builder.Add("present_key = Identity (PresentKey)");
          }

          if (ctx.hasInput(5)) {
            builder.Add("PresentValue = Concat <axis = 2> (past_value, VReshaped)");
          } else {
            builder.Add("PresentValue = Identity (VReshaped)");
          }
          if (ctx.hasOutput(2)) {
            builder.Add("present_value = Identity (PresentValue)");
          }

          if (!defs::nn::utils::AttentionAppendFunctionCausalMask(ctx, builder, true))
            return false;

          // Add padding mask if kv_nonpad_seqlen is provided
          if (ctx.hasInput(6)) {
            builder
                .Add("KVSeqLenExpanded = Unsqueeze(nonpad_kv_seqlen, One1D)") // [batch_size, 1]
                .Add("KVSeqLen0D = Squeeze(KVSeqLen)")
                .Const("Zero0D", static_cast<int64_t>(0))
                .Const("One0D", static_cast<int64_t>(1))
                .Add("Range = Range(Zero0D, KVSeqLen0D, One0D)") // [KVSeqLen,]
                .Add("PaddingMaskBool = Less(Range, KVSeqLenExpanded)") // [batch_size, KVSeqLen]
                .Add("PaddingMaskFloat = Where(PaddingMaskBool, ScalarZero, FloatNegInf)") // [batch_size, KVSeqLen]
                .Add("PaddingMask3D = Unsqueeze(PaddingMaskFloat, One1D)") // [batch_size, 1, KVSeqLen]
                .Add("PaddingMask4D = Unsqueeze(PaddingMask3D, One1D)") // [batch_size, 1, 1, KVSeqLen]
                .Add("AttnBiasCausalPad = Add(AttnBiasCausalOrNot, PaddingMask4D)");
          } else {
            builder.Add("AttnBiasCausalPad = Identity(AttnBiasCausalOrNot)");
          }
          builder.Add("AttnBiasT = Cast (AttnBiasCausalPad)", "to", T1);

          // Group Query Attention is applied if the following are satisfied
          // 1) q_num_heads != kv_num_heads
          // 2) q_num_heads % kv_num_heads == 0
          // 3) kv_num_heads == k_num_heads == v_num_heads
          builder.Add("NGQACond1 = Equal(QNumHeads, KVNumHeads)")
              .Add("GQACond1 = Not(NGQACond1)")
              .Add("DivNumHeads = Div(QNumHeads, KVNumHeads)")
              .Add("IDivNumHeads = Cast(DivNumHeads)", "to", int_type)
              .Add("RemainderNumHeads = Mod(QNumHeads, KVNumHeads)")
              .Add("GQACond2 = Equal(RemainderNumHeads, Zero1D)")
              .Add("GQACond = And(GQACond1, GQACond2)")
              .Add("InterleaveDim = Where(GQACond, IDivNumHeads, One1D)");

          // repeat kv (repeat_interleave)
          builder.Const1D("Two1D", static_cast<int64_t>(2))
              .Add("KUnsqueezed = Unsqueeze(PresentKey, Two1D)") // [B, Hk, 1, T, Dk]
              .Add("VUnsqueezed = Unsqueeze(PresentValue, Two1D)"); // [B, Hk, 1, T, Dv]

          // Build expand shape: [B, Hk, repeats, T, Dk]
          builder
              .Add("KExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, InterleaveDim, NewKVSeqLen, QKHeadSize)")
              .Add("KExpanded = Expand(KUnsqueezed, KExpandShape)");
          builder.Add("VExpandShape = Concat <axis = 0> (BatchSize, KVNumHeads, InterleaveDim, NewKVSeqLen, VHeadSize)")
              .Add("VExpanded = Expand(VUnsqueezed, VExpandShape)");

          // Reshape to [B, Hq, T, Dk] where Hq = Hk * repeats
          builder.Add("KAttentionShape = Concat <axis = 0> (BatchSize, QNumHeads, NewKVSeqLen, QKHeadSize)")
              .Add("VAttentionShape = Concat <axis = 0> (BatchSize, QNumHeads, NewKVSeqLen, VHeadSize)")
              .Add("KAttentionInput = Reshape(KExpanded, KAttentionShape)")
              .Add("VAttentionInput = Reshape(VExpanded, VAttentionShape)");

          // The following pattern is applied
          //      Q          K          V
          //      |          |          |
          //     Q*scale    K*scale     |
          //      |          |          |
          //      |       Transpose     |
          //      |          |          |
          //      ---MatMul---          |
          //            |               |
          // at_mask---Add              |
          //  softcap (if provided)     |
          //            |               |
          //            |               |
          //         Softmax            |
          //            |               |
          //            -----MatMul------
          //                    |
          //                    Y
          builder.Add("KTranspose = Transpose <perm = [0, 1, 3, 2]> (KAttentionInput)")
              .Add("QScaled = Mul(QReshaped, ScaleFactorF)")
              .Add("KScaled = Mul(KTranspose, ScaleFactorF)")
              .Add("QKAttnWeight = MatMul(QScaled, KScaled)")
              .Add("QKAttnCast = Cast (QKAttnWeight)", "to", T1)
              .Add("QKAttnWeightWithBias = Add(QKAttnCast, AttnBiasT)");

          // Apply softcap if provided
          auto softcap_attr = ctx.getAttribute("softcap");
          float softcap_val = (softcap_attr != nullptr) ? softcap_attr->f() : static_cast<float>(0);
          if (softcap_val != 0) {
            builder.Const1D("Softcap", softcap_val)
                .Add("SoftcapF = Cast (Softcap)", "to", T1)
                .Add("SoftcapDiv = Div(QKAttnWeightWithBias, SoftcapF)")
                .Add("SoftcapTanh = Tanh(SoftcapDiv)")
                .Add("QKAttnWeightSoftcap = Mul(SoftcapTanh, SoftcapF)");
          } else {
            builder.Add("QKAttnWeightSoftcap = Identity(QKAttnWeightWithBias)");
          }
          builder.Add("SoftmaxCast = Cast (QKAttnWeightSoftcap)", "to", softmax_precision)
              .Add("AttnWeightSoftmax = Softmax (SoftmaxCast)")
              .Add("SoftmaxOut = Cast (AttnWeightSoftmax)", "to", T1);

          // QK MatMul output if required
          auto qk_matmul_output_mode_attr = ctx.getAttribute("qk_matmul_output_mode");
          int64_t qk_matmul_output_mode = (qk_matmul_output_mode_attr != nullptr) ? qk_matmul_output_mode_attr->i() : 0;
          if (ctx.hasOutput(3)) {
            if (qk_matmul_output_mode == 1) {
              builder.Add("qk_matmul_output = Identity(QKAttnWeightWithBias)");
            } else if (qk_matmul_output_mode == 2) {
              builder.Add("qk_matmul_output = Identity(QKAttnWeightSoftcap)");
            } else if (qk_matmul_output_mode == 3) {
              builder.Add("qk_matmul_output = Identity(AttnWeightSoftmax)");
            } else {
              builder.Add("qk_matmul_output = Identity(QKAttnWeight)");
            }
          }

          builder.Add("YPreReshape = MatMul(SoftmaxOut, VAttentionInput)");
          // Reshape Y to 3D if input is a 3D tensor
          if (is_3d_input) {
            builder.Add("YTranspose = Transpose <perm = [0, 2, 1, 3]> (YPreReshape)")
                .Add("YNewShape = Concat <axis = 0> (Zero1D, Zero1D, NegOne)")
                .Add("Y = Reshape(YTranspose, YNewShape)");
          } else {
            builder.Add("Y = Identity(YPreReshape)");
          }

          schema.BuildFunction(functionProto);
          return true;
        }));
} // namespace ONNX_NAMESPACE
