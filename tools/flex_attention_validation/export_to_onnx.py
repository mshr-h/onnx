#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import (
    flex_attention as fa,
)
from transformers import AttentionInterface, AutoModelForCausalLM, LlamaConfig

import onnx
from onnx import TensorProto, helper


def register_torch_custom_op():
    lib = torch.library.Library("onnx_preview", "DEF")
    lib.define("flex_attention(Tensor q, Tensor k, Tensor v, float alpha) -> Tensor")

    @torch.library.impl(lib, "flex_attention", "CompositeExplicitAutograd")
    def _impl(q, k, v, alpha: float):
        def score_mod(score, batch, head, q_idx, k_idx):
            # 最小の score_mod: score + const
            return score + alpha

        return fa(q, k, v, score_mod=score_mod)

    return lib


def register_transformers_attention_backend(alpha: float):
    def onnx_preview_attention(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        **kwargs,
    ):
        out = torch.ops.onnx_preview.flex_attention(query, key, value, float(alpha))
        return out, None  # attn_weights optional :contentReference[oaicite:12]{index=12}

    AttentionInterface.register(
        "onnx_preview_flex", onnx_preview_attention
    )  # docs :contentReference[oaicite:13]{index=13}


# ---- 3) ONNX postprocess: alpha -> score_mod(GraphProto) ----
def make_score_mod_graph(alpha: float) -> onnx.GraphProto:
    # Inputs follow the FlexAttention score_mod signature idea
    score = helper.make_tensor_value_info("score", TensorProto.FLOAT, [])
    batch = helper.make_tensor_value_info("batch", TensorProto.INT64, [])
    head = helper.make_tensor_value_info("head", TensorProto.INT64, [])
    q_idx = helper.make_tensor_value_info("q_idx", TensorProto.INT64, [])
    k_idx = helper.make_tensor_value_info("k_idx", TensorProto.INT64, [])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [])

    alpha_tensor = helper.make_tensor("alpha", TensorProto.FLOAT, [], [float(alpha)])
    const_node = helper.make_node("Constant", [], ["alpha_const"], value=alpha_tensor)
    add_node = helper.make_node("Add", ["score", "alpha_const"], ["out"])

    g = helper.make_graph(
        nodes=[const_node, add_node],
        name="score_mod_add_const",
        inputs=[score, batch, head, q_idx, k_idx],
        outputs=[out],
        initializer=[],
    )
    return g


def patch_onnx_model(in_path: Path, out_path: Path):
    model = onnx.load(str(in_path))

    for node in model.graph.node:
        if node.domain == "onnx_preview" and node.op_type == "flex_attention":
            # domain/op_type を想定のものへ
            node.domain = "ai.onnx.preview"
            node.op_type = "FlexAttention"

            # alpha 属性を取り出して score_mod(GraphProto) に変換
            alpha = None
            kept_attrs = []
            for a in node.attribute:
                if a.name == "alpha":
                    alpha = a.f
                else:
                    kept_attrs.append(a)

            if alpha is None:
                raise RuntimeError("exported node missing alpha attribute")

            score_mod_graph = make_score_mod_graph(alpha)
            kept_attrs.append(helper.make_attribute("score_mod", score_mod_graph))
            # 置換
            del node.attribute[:]
            node.attribute.extend(kept_attrs)

    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=str, default="artifacts/baseline")
    p.add_argument("--out_onnx", type=str, default="artifacts/flex_attention.onnx")
    p.add_argument("--alpha", type=float, default=0.125)  # score_mod の const
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    baseline_dir = Path(args.baseline_dir)
    out_onnx = Path(args.out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    register_torch_custom_op()
    register_transformers_attention_backend(alpha=args.alpha)

    # baseline から再構築（重みDLなし）
    cfg = LlamaConfig.from_json_file(str(baseline_dir / "config.json"))
    model = AutoModelForCausalLM.from_config(cfg)
    state = torch.load(baseline_dir / "state_dict.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)

    # export 用 backend へ切り替え
    model.set_attn_implementation("onnx_preview_flex")
    model.eval()

    # 入力も baseline と同じものを使用
    inputs = np.load(baseline_dir / "inputs.npz")
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)

    tmp = out_onnx.with_suffix(".tmp.onnx")

    # legacy exporter で custom op を “そのまま” ONNX に落とす（unknown op は残す）
    torch.onnx.export(
        model,
        (input_ids,),
        str(tmp),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        custom_opsets={"onnx_preview": 1},
    )

    # ONNX を編集して score_mod(GraphProto) 属性へ
    patch_onnx_model(tmp, out_onnx)
    tmp.unlink(missing_ok=True)

    print(f"[OK] exported and patched ONNX: {out_onnx.resolve()}")


if __name__ == "__main__":
    main()
