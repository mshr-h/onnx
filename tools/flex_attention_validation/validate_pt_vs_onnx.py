#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

import onnx
from onnx.reference import ReferenceEvaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=str, default="artifacts/baseline")
    p.add_argument("--onnx", type=str, default="artifacts/flex_attention.onnx")
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--verbose", action="store_true", help="ReferenceEvaluator の中間出力を表示（重い）")
    args = p.parse_args()

    baseline_dir = Path(args.baseline_dir)

    # ---- PyTorch ----
    cfg = LlamaConfig.from_json_file(str(baseline_dir / "config.json"))
    model = AutoModelForCausalLM.from_config(cfg)
    state = torch.load(baseline_dir / "state_dict.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    inputs = np.load(baseline_dir / "inputs.npz")
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)

    with torch.no_grad():
        pt_logits = model(input_ids=input_ids, use_cache=False).logits.cpu().numpy()

    # ---- ONNX ReferenceEvaluator ----
    # 文字列パスでも ModelProto でもOK。ここでは明示的に load。
    onx = onnx.load(args.onnx)
    sess = ReferenceEvaluator(onx, verbose=1 if args.verbose else 0)
    # run(None, feeds) は「全出力」を返す :contentReference[oaicite:3]{index=3}
    ort_like_outputs = sess.run(None, {"input_ids": input_ids.numpy()})
    ref_logits = ort_like_outputs[0]  # graph の1つ目の output（logits）

    # ---- Compare ----
    np.testing.assert_allclose(ref_logits, pt_logits, rtol=args.rtol, atol=args.atol)
    max_abs = np.max(np.abs(ref_logits - pt_logits))
    print(f"[OK] allclose passed. max_abs_diff={max_abs:.6g}")


if __name__ == "__main__":
    main()
