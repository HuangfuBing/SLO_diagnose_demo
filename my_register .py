from __future__ import annotations

import os
import re
import json
from typing import Dict, List, Any, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from swift.llm import (
    register_model, ModelMeta, ModelGroup, Model, register_model_arch, MultiModelKeys,
    get_model_tokenizer_with_flash_attn,
    register_template, Template, TemplateMeta, to_float_dtype, get_packed_seq_params
)
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.utils import get_env_args, get_logger

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLConfig

logger = get_logger()

SPM_SPECIAL_TOKEN = "<|SPM_FEAT|>"
SPM_FEAT_DIM = int(os.environ.get("SPM_FEAT_DIM", "768"))
IMAGE_MAX_TOKEN_NUM = int(os.environ.get("IMAGE_MAX_TOKEN_NUM", "1024"))
SPM_STRICT = os.environ.get("SPM_STRICT", "1") != "0"
SPM_DEBUG = os.environ.get("SPM_DEBUG", "0") == "1"

RE_SPM_PATH = re.compile(r"\[SPM_FEAT_PATH\](\S+\.npy)")

register_model_arch(
    MultiModelKeys(
        "my_qwen3_vl_spm",
        language_model=["model.language_model"],
        vision_tower=["model.visual"],
        aligner=["spm_aligner"],
    )
)

def get_model_tokenizer_qwen3_vl_spm(model_dir, *args, **kwargs):
    print("Run my_qwen3_vl_spm...")

    kwargs["automodel_class"] = Qwen3VLForConditionalGeneration
    processor = Qwen3VLProcessor.from_pretrained(model_dir, trust_remote_code=True)

    # NOTE: [MOD] 20251224 Apply MAX_PIXELS / IMG_SIZE budget to processor
    # 2 control ways
    import math
    max_pixels_env = os.getenv("MAX_PIXELS", "1048576")
    img_size_env = os.getenv("IMG_SIZE", "")

    ip = processor.image_processor

    if max_pixels_env:
        mp = int(max_pixels_env)
        ip.max_pixels = mp
        # 可选：同时把 size 限制住（避免配置文件里那种天文数字）
        side = int(math.sqrt(mp))
        ip.size = {"shortest_edge": side, "longest_edge": side}
        print(f"[my_qwen3_vl_spm] apply MAX_PIXELS={mp}, set size≈{side}x{side}")

    elif img_size_env:
        s = int(img_size_env)
        ip.size = {"shortest_edge": s, "longest_edge": s}
        ip.max_pixels = s * s
        print(f"[my_qwen3_vl_spm] apply IMG_SIZE={s}, max_pixels={s*s}")

    print(f"[my_qwen3_vl_spm] image_processor.max_pixels={ip.max_pixels}, size={getattr(ip, 'size', None)}")
    # NOTE: [MOD] end

    # NOTE: [MOD]
    # TODO: make them to env var
    processor.image_processor.disable_grouping = True
    processor.image_processor.min_pixels = 512 * 512

    tokenizer = processor.tokenizer
    added_spm_token = False
    try:
        if (SPM_SPECIAL_TOKEN not in getattr(tokenizer, "additional_special_tokens", [])) and \
           (SPM_SPECIAL_TOKEN not in tokenizer.get_vocab()):
            tokenizer.add_special_tokens({"additional_special_tokens": [SPM_SPECIAL_TOKEN]})
            added_spm_token = True
            print(f"[my_qwen3_vl_spm] add special token: {SPM_SPECIAL_TOKEN}")
    except Exception as e:
        print(f"[my_qwen3_vl_spm][WARN] failed to add special token: {e}")

    kwargs["tokenizer"] = tokenizer
    kwargs["model_config"] = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)

    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model is None:
        return None, None

    if added_spm_token:
        try:
            model.resize_token_embeddings(len(tokenizer))
            print(f"[my_qwen3_vl_spm] resize_token_embeddings -> {len(tokenizer)}")
        except Exception as e:
            print(f"[my_qwen3_vl_spm][WARN] resize_token_embeddings failed: {e}")

    spm_feat_dim = get_env_args("SPM_FEAT_DIM", int, 768)
    hidden_size = getattr(model.config, "hidden_size", None) or model.get_input_embeddings().weight.shape[1]

    dev = model.get_input_embeddings().weight.device
    dt = model.get_input_embeddings().weight.dtype
    model.spm_aligner = nn.Linear(spm_feat_dim, hidden_size, bias=True).to(device=dev, dtype=dt)
    print(f"[my_qwen3_vl_spm] add spm_aligner: {spm_feat_dim} -> {hidden_size} @ {dev}/{dt}")

    native_aligner = None
    if hasattr(model, "model") and hasattr(model.model, "aligner"):
        native_aligner = model.model.aligner
    elif hasattr(model, "aligner"):
        native_aligner = model.aligner
    if native_aligner is not None:
        for p in native_aligner.parameters():
            p.requires_grad = False

    return model, processor


register_model(
    ModelMeta(
        "my_qwen3_vl_spm",
        [
            ModelGroup([
                Model("qwen3-vl-4b-local", "/data1/qwen/model/Qwen3-VL-4B-Instruct"),
                Model("qwen3-vl-32b-local", "/data1/qwen/model/Qwen3-VL-32B-Instruct"),
            ])
        ],
        "my_qwen3_vl_spm",
        get_model_tokenizer_qwen3_vl_spm,
        is_multimodal=True,
        model_arch="my_qwen3_vl_spm",
        architectures=["Qwen3VLForConditionalGeneration"],
        requires=["transformers>=4.57"],
        tags=["vision", "medical"],
        additional_saved_files=["my_spk_dict.pt"],
    )
)

# NOTE: [MOD] fucking wrapper
class _VisualTensorOnlyWrapper:
    """Wrap a vision nn.Module so that calling it always returns a Tensor (unwrap tuple/list outputs),
    while keeping attributes like .dtype available for Swift base code.
    """
    def __init__(self, visual):
        self.visual = visual

    @property
    def dtype(self):
        # Some modules may not expose .dtype directly; fall back to first parameter
        if hasattr(self.visual, "dtype"):
            return self.visual.dtype
        try:
            return next(self.visual.parameters()).dtype
        except Exception:
            return torch.float32

    @property
    def device(self):
        if hasattr(self.visual, "device"):
            return self.visual.device
        try:
            return next(self.visual.parameters()).device
        except Exception:
            return torch.device("cpu")

    def __getattr__(self, name):
        # Delegate everything else to the wrapped visual module
        return getattr(self.visual, name)

    def __call__(self, *args, **kwargs):
        out = self.visual(*args, **kwargs)

        # tuple: (tensor, aux) or (tensor, [tensor...])
        if isinstance(out, tuple):
            if len(out) > 0 and torch.is_tensor(out[0]):
                return out[0]
            # find first tensor anywhere
            for x in out:
                if torch.is_tensor(x):
                    return x
                if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
                    return x[0]
            raise TypeError(f"[SPM] visual() returned tuple but no Tensor found: {type(out)}")

        # list[tensor]
        if isinstance(out, list):
            if len(out) > 0 and torch.is_tensor(out[0]):
                # try concat along token dim, fallback to first
                try:
                    return torch.cat(out, dim=0)
                except Exception:
                    return out[0]
            raise TypeError(f"[SPM] visual() returned list but not tensor list: {type(out)}")

        # ModelOutput-like
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            return out.last_hidden_state

        # already Tensor
        if torch.is_tensor(out):
            return out

        raise TypeError(f"[SPM] visual() returned unsupported type: {type(out)}")

class Qwen3VLSPMTemplate(Template):
    use_model = True
    # NOTE: XXX: not support False.
    support_padding_free = True
    norm_bbox = "none"

    placeholder_tokens = ["<|image_pad|>", "<|IMAGE|>", SPM_SPECIAL_TOKEN]

    def _get_image_token_str(self) -> str:
        tk = getattr(self.processor, "tokenizer", None)
        if tk is None:
            return "<|image_pad|>"
        for cand in ("<|image_pad|>", "<|IMAGE|>"):
            try:
                if cand in tk.get_vocab() or cand in getattr(tk, "additional_special_tokens", []):
                    return cand
            except Exception:
                pass
        cfg = getattr(self.model_info, "config", None)
        image_token_id = getattr(cfg, "image_token_id", None)
        if image_token_id is not None:
            try:
                tok = tk.convert_ids_to_tokens(int(image_token_id))
                if isinstance(tok, str) and tok:
                    return tok
            except Exception:
                pass
        return "<|image_pad|>"

    def replace_tag(self, media_type: Literal["image"], index: int, inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == "image"
        img_tok = self._get_image_token_str()
        return [f"<|vision_start|>{img_tok}<|vision_end|>"]

    def _get_messages_anyway(self, inputs: StdTemplateInputs):
        msgs = getattr(inputs, "messages", None)
        if isinstance(msgs, list) and msgs:
            return msgs
        raw = getattr(inputs, "raw", None) or {}
        msgs = raw.get("messages", None)
        if isinstance(msgs, list) and msgs:
            return msgs
        return None

    def _parse_spm_path_anyway(self, inputs: StdTemplateInputs) -> Optional[str]:
        msgs = self._get_messages_anyway(inputs)
        if not isinstance(msgs, list):
            return None
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = m.get("content", "")
            if not isinstance(c, str):
                continue
            mm = RE_SPM_PATH.search(c)
            if mm:
                return mm.group(1)
        return None

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)

        processor = self.processor
        media_inputs = processor(
            text="",
            images=inputs.images or None,
            return_tensors="pt",
            do_resize=True,
        )
        media_inputs.pop("input_ids", None)
        media_inputs.pop("attention_mask", None)
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)

        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale")

        image_token_ids = self._tokenize(self._get_image_token_str())
        idx_list = findall(input_ids, image_token_ids)
        if idx_list:
            image_grid_thw = media_inputs.get("image_grid_thw")
            merge_size = processor.image_processor.merge_size

            def _get_new_tokens(i: int):
                token_len = int(image_grid_thw[i].prod().item() // (merge_size ** 2))
                if IMAGE_MAX_TOKEN_NUM > 0:
                    token_len = min(token_len, IMAGE_MAX_TOKEN_NUM)
                return image_token_ids * token_len

            input_ids, labels, loss_scale = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens
            )

        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        encoded["loss_scale"] = loss_scale
        encoded.update(media_inputs)

        spm_path = self._parse_spm_path_anyway(inputs)
        raw = getattr(inputs, "raw", None) or {}
        sample_id = raw.get("id", raw.get("image", raw.get("path", None)))

        if SPM_DEBUG and not hasattr(self, "_dbg_encode_n"):
            self._dbg_encode_n = 0
        if SPM_DEBUG and self._dbg_encode_n < 3:
            self._dbg_encode_n += 1
            print(f"[SPM][_encode] sample_id={sample_id} spm_path={spm_path} keys={list(encoded.keys())[:12]}")

        if not spm_path: # NOTE: spm_path could be found
            msg = f"[my_qwen3_vl_spm] cannot find [SPM_FEAT_PATH]...npy in messages. sample_id={sample_id}"
            if SPM_STRICT:
                raise ValueError(msg)
            logger.warning(msg)
            return encoded

        if not os.path.isabs(spm_path):
            spm_path = os.path.abspath(spm_path)
        if not os.path.exists(spm_path):
            msg = f"[my_qwen3_vl_spm] spm npy not found: {spm_path} sample_id={sample_id}"
            if SPM_STRICT:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            return encoded

        arr = np.load(spm_path)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.shape[0] != SPM_FEAT_DIM:
            msg = f"[my_qwen3_vl_spm] spm dim mismatch: got {arr.shape[0]} expect {SPM_FEAT_DIM} path={spm_path} sample_id={sample_id}"
            if SPM_STRICT:
                raise ValueError(msg)
            logger.warning(msg)
            return encoded

        encoded["spm_feats"] = torch.from_numpy(arr)  # [D]
        encoded["__spm_path"] = spm_path
        encoded["__sample_id"] = sample_id
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "spm_feats" not in inputs:
            raise ValueError(f"[my_qwen3_vl_spm] spm_feats missing in _post_encode. inputs_keys={list(inputs.keys())}")

        if SPM_DEBUG and not hasattr(self, "_dbg_post_n"):
            self._dbg_post_n = 0
        if SPM_DEBUG and self._dbg_post_n < 3:
            self._dbg_post_n += 1
            print(f"[SPM][_post_encode] keys={list(inputs.keys())[:20]} spm_feats_shape={tuple(inputs['spm_feats'].shape)}")

        base_model = self.get_base_model(model)

        input_ids = inputs.get("input_ids")
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=base_model.get_input_embeddings().weight.device)
        elif isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        inputs["input_ids"] = input_ids

        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        # 1) text embeds
        embed_layer = base_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        # 2) 融合视觉
        # print("ENV IMAGE_MAX_TOKEN_NUM = ", os.getenv("IMAGE_MAX_TOKEN_NUM"))
        # print("ENV MAX_PIXELS = ", os.getenv("MAX_PIXELS"))
        inputs_embeds = self._get_inputs_embeds_hf(
            inputs_embeds,
            inputs,
            model.visual,
            self.processor,
            model.config,
        )

        # 3) 注入 SPM：定位 spans
        spm_token_ids = self._tokenize(SPM_SPECIAL_TOKEN)

        # NOTE: [MOD] 20251224 findall spm token: robust for input_ids being [B, L] -> tolist becomes [[...]]
        # Return idx_list as List[List[Tuple[int,int]]] with (start, end) end-exclusive, per sample.
        def _find_spans_batch(ids: torch.Tensor, pat: List[int]) -> List[List[Tuple[int, int]]]:
            if not isinstance(ids, torch.Tensor):
                raise TypeError("input_ids must be torch.Tensor")
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            elif ids.dim() != 2:
                raise ValueError(f"input_ids must be [L] or [B,L], got {tuple(ids.shape)}")

            B, L = ids.shape
            K = len(pat)
            spans: List[List[Tuple[int, int]]] = [[] for _ in range(B)]
            if K <= 0:
                return spans

            # fast path: single token id
            if K == 1:
                tid = int(pat[0])
                pos = (ids == tid).nonzero(as_tuple=False)  # [N,2] -> (b, idx)
                for b, idx in pos.tolist():
                    spans[b].append((idx, idx + 1))  # end-exclusive
                return spans

            # general path: multi-token span match using unfold (no tolist)
            if L < K:
                return spans
            pat_t = torch.tensor(pat, device=ids.device, dtype=ids.dtype)  # [K]
            windows = ids.unfold(dimension=1, size=K, step=1)             # [B, L-K+1, K]
            matches = (windows == pat_t).all(dim=-1)                      # [B, L-K+1]
            pos = matches.nonzero(as_tuple=False)                         # [N,2] -> (b, start)
            for b, start in pos.tolist():
                spans[b].append((start, start + K))  # end-exclusive
            return spans

        idx_list = _find_spans_batch(input_ids, spm_token_ids)
        has_any = any(len(v) > 0 for v in idx_list)
        if not has_any:
            raise ValueError("[my_qwen3_vl_spm] cannot find <|SPM_FEAT|> span in prompt.")
        # NOTE: [MOD] end

        spm_feats = inputs["spm_feats"]
        if isinstance(spm_feats, torch.Tensor) and spm_feats.dim() == 1:
            spm_feats = spm_feats.unsqueeze(0)
        elif not isinstance(spm_feats, torch.Tensor):
            spm_feats = torch.tensor(spm_feats, dtype=torch.float32)

        spm_feats = spm_feats.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        spm_proj = model.spm_aligner(spm_feats)  # [B,H]

        # NOTE: [MOD] 20251225 support packed-seq batch for SPM injection, find spm token to cut batches
        B_model = input_ids.shape[0]
        B_spm = spm_proj.shape[0]

        def _find_spans_1d(ids_1d: torch.Tensor, pat: List[int]) -> List[Tuple[int, int]]:
            """Return spans (start, end) end-exclusive in a 1D token tensor."""
            if ids_1d.dim() != 1:
                ids_1d = ids_1d.view(-1)
            L = ids_1d.numel()
            K = len(pat)
            spans: List[Tuple[int, int]] = []
            if K <= 0 or L < K:
                return spans
            if K == 1:
                tid = int(pat[0])
                pos = (ids_1d == tid).nonzero(as_tuple=False).view(-1)
                for p in pos.tolist():
                    spans.append((p, p + 1))
                return spans
            pat_t = torch.tensor(pat, device=ids_1d.device, dtype=ids_1d.dtype)
            windows = ids_1d.unfold(dimension=0, size=K, step=1)   # [L-K+1, K]
            matches = (windows == pat_t).all(dim=-1)               # [L-K+1]
            pos = matches.nonzero(as_tuple=False).view(-1)
            for p in pos.tolist():
                spans.append((p, p + K))
            return spans

        def _get_cu_seqlens(inp: Dict[str, Any]) -> Optional[torch.Tensor]:
            # try common keys from packed-seq toolchains
            for k in ("cu_seqlens", "cu_seqlens_q", "cu_seqlens_k", "cu_seqlens_padded"):
                v = inp.get(k, None)
                if isinstance(v, torch.Tensor) and v.numel() >= 2:
                    return v
            return None

        if B_model == B_spm:
            # normal un-packed batch: [B, L]
            for b in range(B_model):
                for start, end in idx_list[b]:
                    inputs_embeds[b, start:end, :] = spm_proj[b].unsqueeze(0).expand(end - start, -1)

        elif B_model == 1 and B_spm > 1:
            # packed batch: multiple samples -> one long seq, input_ids: [1, L_total]
            ids1 = input_ids[0]  # [L_total]
            cu = _get_cu_seqlens(inputs)

            injected = 0
            if cu is not None:
                cu = cu.to(device=ids1.device).flatten()
                # We expect cu_seqlens length == B_spm + 1 (prefix sums)
                if cu.numel() == B_spm + 1:
                    for i in range(B_spm):
                        s0 = int(cu[i].item())
                        e0 = int(cu[i + 1].item())
                        spans_local = _find_spans_1d(ids1[s0:e0], spm_token_ids)
                        if not spans_local:
                            raise ValueError(
                                f"[my_qwen3_vl_spm] packed-seq: cannot find <|SPM_FEAT|> in segment {i} "
                                f"(range {s0}:{e0})."
                            )
                        for (ls, le) in spans_local:
                            gs, ge = s0 + ls, s0 + le
                            inputs_embeds[0, gs:ge, :] = spm_proj[i].unsqueeze(0).expand(ge - gs, -1)
                        injected += 1

            if injected == 0:
                # fallback: assign by occurrence order in the packed sequence
                spans_all = _find_spans_1d(ids1, spm_token_ids)
                if len(spans_all) < B_spm:
                    raise ValueError(
                        f"[my_qwen3_vl_spm] packed-seq fallback: found {len(spans_all)} SPM spans, "
                        f"but spm_feats batch is {B_spm}. Need >= {B_spm}."
                    )
                for i in range(B_spm):
                    s, e = spans_all[i]
                    inputs_embeds[0, s:e, :] = spm_proj[i].unsqueeze(0).expand(e - s, -1)

        else:
            raise ValueError(
                f"[my_qwen3_vl_spm] unsupported batch layout: input_ids batch={B_model}, spm_feats batch={B_spm}. "
                f"(Maybe padding-free packing behavior changed.)"
            )
        # [MOD] ---- end ----

        # 4) position_ids/rope_deltas（Qwen3-VL XOR 约束）
        try:
            position_ids, rope_deltas = base_model.model.get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=inputs.get("image_grid_thw", None),
            )
        except TypeError:
            position_ids, rope_deltas = base_model.model.get_rope_index(
                input_ids, attention_mask, inputs.get("image_grid_thw", None)
            )

        out: Dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            # FIXME: "rope_deltas": rope_deltas,
        }
        if "labels" in inputs:
            out["labels"] = inputs["labels"]
            if "loss_scale" in inputs:
                out["loss_scale"] = inputs["loss_scale"]
        else: # without labels means eval mode
            out["input_ids"] = input_ids
            
        """
        if "labels" in inputs:
            out["labels"] = inputs["labels"]
        if "loss_scale" in inputs:
            out["loss_scale"] = inputs["loss_scale"]
        """
        return out

    # NOTE: 20251222-modify
    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        bad = []
        spm_list = []  # [MOD] 提前准备缓存

        for i, ex in enumerate(batch):
            if not isinstance(ex, dict):
                bad.append({"i": i, "reason": "non-dict", "type": str(type(ex))})
                continue

            if "spm_feats" not in ex:
                bad.append({
                    "i": i,
                    "reason": "missing_key",
                    "sample_id": ex.get("__sample_id", None),
                    "spm_path": ex.get("__spm_path", None),
                    "keys": list(ex.keys())[:40],
                })
                continue

            t = ex.get("spm_feats", None)
            if t is None:
                bad.append({
                    "i": i,
                    "reason": "spm_feats_is_none",
                    "sample_id": ex.get("__sample_id", None),
                    "spm_path": ex.get("__spm_path", None),
                    "keys": list(ex.keys())[:40],
                })
                continue

            try:
                tt = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                tt = tt.float().reshape(-1)
            except Exception as e:
                bad.append({
                    "i": i,
                    "reason": "cannot_tensorize",
                    "sample_id": ex.get("__sample_id", None),
                    "spm_path": ex.get("__spm_path", None),
                    "type": str(type(t)),
                    "exc": repr(e),
                })
                continue

            if tt.numel() != SPM_FEAT_DIM:
                bad.append({
                    "i": i,
                    "reason": "dim_mismatch",
                    "sample_id": ex.get("__sample_id", None),
                    "spm_path": ex.get("__spm_path", None),
                    "numel": int(tt.numel()),
                    "expect": int(SPM_FEAT_DIM),
                })
                continue

            spm_list.append(tt)  # [MOD] super() 前就缓存下来

        if bad:
            # 你原来的打印逻辑可以保留
            print("[my_qwen3_vl_spm][COLLATOR] bad samples (spm_feats problem):")
            for item in bad[:6]:
                i = item["i"]
                ex = batch[i] if isinstance(batch[i], dict) else {}
                msgs = ex.get("messages", None)
                if msgs is None:
                    raw = ex.get("raw", None)
                    if isinstance(raw, dict):
                        msgs = raw.get("messages", None)
                preview = ""
                if isinstance(msgs, list):
                    parts = []
                    for mm in msgs[:3]:
                        if isinstance(mm, dict):
                            parts.append(str(mm.get("content", "")))
                    preview = " | ".join(parts).replace("\n", "\\n")[:300]
                item["messages_preview"] = preview
                print(json.dumps(item, ensure_ascii=False))
            raise ValueError("[my_qwen3_vl_spm] spm_feats invalid. See printed bad samples above.")

        # [MOD] 先跑 super() 得到标准 batch
        res = super()._data_collator(batch, padding_to=padding_to)

        # [MOD] 再把 spm_feats 塞回 res（不再访问 batch）
        res["spm_feats"] = torch.stack(spm_list, dim=0)

        if SPM_DEBUG and not hasattr(self, "_dbg_collate_n"):
            self._dbg_collate_n = 0
        if SPM_DEBUG and self._dbg_collate_n < 3:
            self._dbg_collate_n += 1
            print(f"[SPM][_collator] spm_feats={tuple(res['spm_feats'].shape)} res_keys={list(res.keys())[:20]}")

        if "text_position_ids" in res:
            res.update(get_packed_seq_params(res["text_position_ids"]))

        return res

    # NOTE: [MOD] 20251224
    def _get_inputs_embeds_hf(self, inputs_embeds, inputs, visual, processor, config, **kwargs):
        visual_wrapped = _VisualTensorOnlyWrapper(visual)
        return super()._get_inputs_embeds_hf(inputs_embeds, inputs, visual_wrapped, processor, config, **kwargs)

register_template(
    TemplateMeta(
        template_type="my_qwen3_vl_spm",
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>"],
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
        default_system="",
        stop_words=["<|endoftext|>"],
        agent_template="hermes",
        template_cls=Qwen3VLSPMTemplate,
    )
)

print("[my_qwen3_vl_spm] register.py loaded: model_type=my_qwen3_vl_spm, template_type=my_qwen3_vl_spm, template_cls=Qwen3VLSPMTemplate")
