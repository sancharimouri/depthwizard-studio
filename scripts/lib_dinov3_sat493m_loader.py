"""Load Meta's DINOv3 ViT-L/16 SAT493M backbone from the Hugging Face
checkpoint (facebook/dinov3-vitl16-pretrain-sat493m, downloaded with an
approved gated-access token) and wire it into facebookresearch/dinov3's own
CHMv2 DPT depth head, unmodified.

The HF repo packages these weights under `transformers`' own DINOv3 port,
whose parameter names (`layer.N.attention.q_proj`, `mlp.up_proj`, ...) don't
match the original research repo's naming (`blocks.N.attn.qkv`,
`mlp.fc1`, ...) -- same pretrained weights, same architecture, two
independent re-implementations. This module is a pure state-dict key/shape
translation between the two, not a re-derivation or a custom readout: no
weight values are computed or changed, only renamed/reshaped (the fused qkv
projection is a concatenation of HF's split q/k/v projections).

`local_cls_norm` (SAT493M's untied global/local cls-token norm) is a
training-only branch (see vision_transformer.py: only exercised when
`self.training and idx == 1`) and is never used in eval()/no_grad
inference, so it's intentionally left out of both the checkpoint and the
constructed model (`untie_global_and_local_cls_norm=False`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DINOV3_REPO = PROJECT_ROOT / "external" / "dinov3"
if str(DINOV3_REPO) not in sys.path:
    sys.path.insert(0, str(DINOV3_REPO))

from dinov3.eval.depth.models import make_depther_from_config  # noqa: E402
from dinov3.hub.depthers import _get_chmv2_config  # noqa: E402
from dinov3.hub.backbones import dinov3_vitl16  # noqa: E402
from dinov3.hub.utils import _DINOV3_BASE_URL, _safe_load_state_dict_from_url  # noqa: E402

N_LAYERS = 24


def convert_hf_state_dict(hf_state: dict) -> dict:
    out = {}
    out["cls_token"] = hf_state["embeddings.cls_token"]
    out["storage_tokens"] = hf_state["embeddings.register_tokens"]
    out["mask_token"] = hf_state["embeddings.mask_token"].squeeze(0)
    out["patch_embed.proj.weight"] = hf_state["embeddings.patch_embeddings.weight"]
    out["patch_embed.proj.bias"] = hf_state["embeddings.patch_embeddings.bias"]
    out["norm.weight"] = hf_state["norm.weight"]
    out["norm.bias"] = hf_state["norm.bias"]

    for i in range(N_LAYERS):
        p = f"layer.{i}."
        b = f"blocks.{i}."

        out[b + "norm1.weight"] = hf_state[p + "norm1.weight"]
        out[b + "norm1.bias"] = hf_state[p + "norm1.bias"]
        out[b + "norm2.weight"] = hf_state[p + "norm2.weight"]
        out[b + "norm2.bias"] = hf_state[p + "norm2.bias"]

        out[b + "attn.qkv.weight"] = torch.cat(
            [hf_state[p + "attention.q_proj.weight"], hf_state[p + "attention.k_proj.weight"], hf_state[p + "attention.v_proj.weight"]],
            dim=0,
        )
        # k_proj has no bias in this checkpoint (q/v do); the target's fused
        # qkv linear has one bias covering all three, so the k-segment is
        # zeros -- equivalent to "no bias" for k, not a missing value.
        q_bias = hf_state[p + "attention.q_proj.bias"]
        k_bias = torch.zeros_like(q_bias)
        v_bias = hf_state[p + "attention.v_proj.bias"]
        out[b + "attn.qkv.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)
        out[b + "attn.proj.weight"] = hf_state[p + "attention.o_proj.weight"]
        out[b + "attn.proj.bias"] = hf_state[p + "attention.o_proj.bias"]

        # LinearKMaskedBias's mask: q and v bias pass through (1), k bias is
        # zeroed (0) -- a fixed, deterministic pattern (mask_k_bias=True),
        # not learned data, matching how the target model derives it itself.
        dim = q_bias.numel()
        out[b + "attn.qkv.bias_mask"] = torch.cat(
            [torch.ones(dim), torch.zeros(dim), torch.ones(dim)]
        )

        out[b + "ls1.gamma"] = hf_state[p + "layer_scale1.lambda1"]
        out[b + "ls2.gamma"] = hf_state[p + "layer_scale2.lambda1"]

        out[b + "mlp.fc1.weight"] = hf_state[p + "mlp.up_proj.weight"]
        out[b + "mlp.fc1.bias"] = hf_state[p + "mlp.up_proj.bias"]
        out[b + "mlp.fc2.weight"] = hf_state[p + "mlp.down_proj.weight"]
        out[b + "mlp.fc2.bias"] = hf_state[p + "mlp.down_proj.bias"]

    return out


def load_sat493m_backbone(hf_safetensors_path: Path) -> torch.nn.Module:
    hf_state = load_file(str(hf_safetensors_path))
    converted = convert_hf_state_dict(hf_state)

    backbone = dinov3_vitl16(pretrained=False)
    missing, unexpected = backbone.load_state_dict(converted, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys after conversion: {unexpected}")
    # rope_embed.periods is a deterministic buffer computed by
    # RopePositionEmbedding.__init__ -> _init_weights() from the model's own
    # config (base/period, head dim), not learned data -- already correct
    # immediately after dinov3_vitl16(pretrained=False) construction.
    allowed_missing = {"rope_embed.periods"}
    real_missing = [k for k in missing if k not in allowed_missing]
    if real_missing:
        raise RuntimeError(f"Missing keys after conversion: {real_missing}")

    return backbone


def load_dinov3_chmv2_depther(hf_safetensors_path: Path) -> torch.nn.Module:
    """SAT493M backbone (from the HF checkpoint) + Meta's official, unmodified
    CHMv2 DPT depth head (downloaded from the public fbaipublicfiles URL)."""
    backbone = load_sat493m_backbone(hf_safetensors_path)

    depther = make_depther_from_config(backbone, config=_get_chmv2_config())

    head_url = _DINOV3_BASE_URL + "/chmv2/dinov3_vitl16_chmv2_dpt_head-3703d643.pth"
    checkpoint = _safe_load_state_dict_from_url(head_url, map_location="cpu", check_hash=False)
    depther.decoder.load_state_dict(checkpoint, strict=True)

    depther.eval()
    return depther
