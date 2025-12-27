```sh
# 1) baseline 作成（flex_attention backend で実行）
python tools/flex_attention_validation/make_baseline_model.py \
  --out_dir artifacts/baseline --seq_len 8

# 2) export（置換 + score_mod(GraphProto) まで作る）
python tools/flex_attention_validation/export_to_onnx.py \
  --baseline_dir artifacts/baseline --out_onnx artifacts/flex_attention.onnx

# 3) validate（FlexAttention が ONNX/ORT 側で動くようになってから）
python tools/flex_attention_validation/validate_pt_vs_onnx.py \
  --baseline_dir artifacts/baseline --onnx artifacts/flex_attention.onnx
```
