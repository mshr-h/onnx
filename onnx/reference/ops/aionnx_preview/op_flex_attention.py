from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from onnx.reference.op_run import OpRun


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    # x: (S,) float32
    m = np.max(x)
    ex = np.exp(x - m)
    s = np.sum(ex)
    if s == 0:
        # All -inf (or very negative). Spec doesn't define; return zeros.
        return np.zeros_like(x)
    return ex / s


def _to_scalar_tensor_i64(v: int) -> np.ndarray:
    # Indices are scalar tensors (0-d arrays)
    return np.array(v, dtype=np.int64)


def _to_scalar_tensor_f32(v: float) -> np.ndarray:
    return np.array(v, dtype=np.float32)


class FlexAttention(OpRun):
    """
    Reference implementation for ai.onnx.preview::FlexAttention (preview, v1).
    """

    op_domain = "ai.onnx.preview"

    def _call_graph_attr(
        self,
        evaluator: Any,
        positional_inputs: Sequence[np.ndarray],
        *,
        attributes: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        input_names = list(evaluator.input_names)
        if len(input_names) != len(positional_inputs):
            raise RuntimeError(
                f"Graph attribute expects {len(input_names)} inputs "
                f"but got {len(positional_inputs)}."
            )
        feeds = dict(zip(input_names, positional_inputs, strict=False))
        outs = evaluator.run(None, feeds, attributes=attributes)
        if not isinstance(outs, list) or len(outs) != 1:
            raise RuntimeError("Graph attribute must produce exactly 1 output.")
        return outs[0]

    def _run(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        *mod_inputs: np.ndarray,
        scale: Optional[float] = None,
        enable_gqa: Optional[int] = None,
        mask_value: Optional[float] = None,
        score_mod: Any = None,
        mask_mod: Any = None,
        prob_mod: Any = None,
        **_: Any,
    ) -> Tuple[np.ndarray]:
        """Inputs:
          query: (B, Hq, L, E)
          key  : (B, Hkv, S, E)
          value: (B, Hkv, S, Ev)
          mod_inputs: extra tensors forwarded to graphs (optional)

        Attributes (all optional):
          scale, enable_gqa, mask_value, score_mod, mask_mod, prob_mod
        """
        # Attributes may already be loaded onto self by OpRun._load_attributes().
        # Prefer self.<attr> if present.
        score_mod = getattr(self, "score_mod", score_mod)
        mask_mod = getattr(self, "mask_mod", mask_mod)
        prob_mod = getattr(self, "prob_mod", prob_mod)
        enable_gqa = int(getattr(self, "enable_gqa", 0 if enable_gqa is None else enable_gqa))
        scale = getattr(self, "scale", scale)
        mask_value = getattr(self, "mask_value", mask_value)

        if mask_value is None:
            # "approx -inf" default (float32 min)
            mask_value = float(np.finfo(np.float32).min)
        if scale is None:
            # NOTE: exporter SHOULD set scale explicitly; reference treats None as identity.
            scale = 1.0

        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise RuntimeError("query/key/value must be rank-4 tensors.")

        B, Hq, L, E = query.shape
        Bk, Hkv, S, Ek = key.shape
        Bv, Hkv2, Sv, Ev = value.shape

        if Bk != B or Bv != B:
            raise RuntimeError("Batch dimension mismatch among query/key/value.")
        if Ek != E:
            raise RuntimeError("Embedding dim mismatch between query and key.")
        if Hkv2 != Hkv or Sv != S:
            raise RuntimeError("key/value head or sequence length mismatch.")
        if enable_gqa == 0 and Hq != Hkv:
            raise RuntimeError("enable_gqa=0 requires Hq == Hkv.")
        if enable_gqa == 1 and (Hq % Hkv) != 0:
            raise RuntimeError("enable_gqa=1 requires Hq % Hkv == 0.")

        # Compute in float32 for stability/portability.
        q_f = query.astype(np.float32, copy=False)
        k_f = key.astype(np.float32, copy=False)
        v_f = value.astype(np.float32, copy=False)

        group = (Hq // Hkv) if enable_gqa == 1 else 1

        out = np.empty((B, Hq, L, Ev), dtype=np.float32)

        # Main loops: b, hq, q_idx
        for b in range(B):
            for hq in range(Hq):
                kvh = (hq // group) if enable_gqa == 1 else hq

                # Slice K,V once per head for locality
                K_mat = k_f[b, kvh, :, :]  # (S, E)
                V_mat = v_f[b, kvh, :, :]  # (S, Ev)

                for q_idx in range(L):
                    q_vec = q_f[b, hq, q_idx, :]  # (E,)

                    # raw scores: (S,)
                    scores = (K_mat @ q_vec) * float(scale)  # float32

                    # Apply score_mod / mask_mod elementwise if provided
                    if score_mod is not None or mask_mod is not None:
                        scores2 = np.empty((S,), dtype=np.float32)
                        for kv_idx in range(S):
                            s = float(scores[kv_idx])

                            if score_mod is not None:
                                # (score, batch, head, q_idx, kv_idx, *mod_inputs) -> score_out
                                pos = [
                                    _to_scalar_tensor_f32(s),
                                    _to_scalar_tensor_i64(b),
                                    _to_scalar_tensor_i64(hq),
                                    _to_scalar_tensor_i64(q_idx),
                                    _to_scalar_tensor_i64(kv_idx),
                                    *mod_inputs,
                                ]
                                s_out = self._call_graph_attr(score_mod, pos)
                                s = float(np.asarray(s_out, dtype=np.float32).reshape(()))

                            if mask_mod is not None:
                                # (batch, head, q_idx, kv_idx, *mod_inputs) -> bool
                                pos = [
                                    _to_scalar_tensor_i64(b),
                                    _to_scalar_tensor_i64(hq),
                                    _to_scalar_tensor_i64(q_idx),
                                    _to_scalar_tensor_i64(kv_idx),
                                    *mod_inputs,
                                ]
                                m_out = self._call_graph_attr(mask_mod, pos)
                                m = bool(np.asarray(m_out).reshape(()))
                                if not m:
                                    s = float(mask_value)

                            scores2[kv_idx] = np.float32(s)
                        scores = scores2

                    # Softmax over kv dimension
                    probs = _softmax_1d(scores)

                    # Apply prob_mod elementwise if provided
                    if prob_mod is not None:
                        probs2 = np.empty((S,), dtype=np.float32)
                        for kv_idx in range(S):
                            p = float(probs[kv_idx])
                            # (prob, batch, head, q_idx, kv_idx, *mod_inputs) -> prob_out
                            pos = [
                                _to_scalar_tensor_f32(p),
                                _to_scalar_tensor_i64(b),
                                _to_scalar_tensor_i64(hq),
                                _to_scalar_tensor_i64(q_idx),
                                _to_scalar_tensor_i64(kv_idx),
                                *mod_inputs,
                            ]
                            p_out = self._call_graph_attr(prob_mod, pos)
                            p = float(np.asarray(p_out, dtype=np.float32).reshape(()))
                            probs2[kv_idx] = np.float32(p)
                        probs = probs2
                        # NOTE: Spec案どおり「自動再正規化はしない」
                        # 必要なら prob_mod 側で実装する。

                    # Y = probs @ V
                    out[b, hq, q_idx, :] = probs @ V_mat  # (Ev,)

        # Cast back to value dtype (or query dtype). v1では T として揃える想定。
        y = out.astype(value.dtype, copy=False)
        return (y,)
