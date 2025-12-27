#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, LlamaConfig


def pick_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="artifacts/baseline")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)

    # tiny config
    p.add_argument("--vocab_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--intermediate_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=1)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)

    # Llama の極小 config（重みDLなし）
    cfg = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=max(64, args.seq_len),
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )

    # まず普通に作ってから backend を切り替える（from_config が attn_implementation を受けない版にも対応）
    model = AutoModelForCausalLM.from_config(cfg)
    try:
        model.set_attn_implementation("flex_attention")  # docs 記載あり :contentReference[oaicite:7]{index=7}
    except Exception as e:
        raise RuntimeError(
            "この transformers では set_attn_implementation('flex_attention') が失敗しました。"
            " transformers/torch の組合せを見直してください。"
        ) from e

    model.eval().to(device)

    # 入力（重みDLしないのでランダム token）
    input_ids = torch.randint(
        low=0, high=args.vocab_size, size=(args.batch, args.seq_len), dtype=torch.long, device=device
    )

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits.detach().cpu().numpy()

    # 保存（state_dict を保存するのが A の肝）
    (out_dir / "config.json").write_text(cfg.to_json_string(), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "state_dict.pt")
    np.savez(out_dir / "inputs.npz", input_ids=input_ids.detach().cpu().numpy())
    np.savez(out_dir / "pt_outputs.npz", logits=logits)

    meta = {
        "seed": args.seed,
        "device": str(device),
        "attn_implementation": "flex_attention",
        "shape_input_ids": list(input_ids.shape),
        "shape_logits": list(logits.shape),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] wrote baseline artifacts to: {out_dir.resolve()}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
