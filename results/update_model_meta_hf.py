# ruff: noqa

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _get_max_tokens_from_config(cfg) -> Optional[int]:
    # Try common HF config attributes
    for attr in [
        "max_position_embeddings",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    ]:
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return int(val)
    # Some SentenceTransformer wrappers store this under model config
    # but if not found, fall back to 512 (BERT default)
    return 512


def _get_embed_dim_from_config(cfg) -> Optional[int]:
    # Try common config attributes that indicate embedding/hidden dimension
    if cfg is None:
        return None
    # Direct attributes
    for attr in [
        "sentence_embedding_dimension",  # Sentence-Transformers sometimes expose this
        "embed_dim",  # some models
        "embedding_size",  # generic
        "hidden_size",  # BERT-like
        "d_model",  # T5/Transformer encoder
        "projection_dim",  # CLIP-like projection
        "word_embed_proj_dim",  # some LLaMA variants
        "dim",  # generic configs
    ]:
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return int(val)
    # Nested text_config (e.g., CLIP-like)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for attr in ["projection_dim", "hidden_size", "embed_dim", "d_model"]:
            val = getattr(text_cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return int(val)
    return None


def _count_parameters_hf(model) -> int:
    try:
        import torch  # noqa
    except Exception:
        # Fallback: sum .numel via parameters iterator even if torch not available (rare)
        total = 0
        for p in model.parameters():  # type: ignore[attr-defined]
            try:
                total += int(p.numel())
            except Exception:
                pass
        return int(total)
    else:
        total = sum(p.numel() for p in model.parameters())  # type: ignore[attr-defined]
        return int(total)


def compute_hf_meta(
    model_id: str,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Load a Hugging Face model/config and return (n_parameters, max_tokens, embed_dim).

    We try to avoid full weight allocation when possible by using AutoConfig first,
    but to count parameters we need the model; we load it in no_grad and eval mode.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModel
    except Exception as e:
        print(f"transformers (and torch) are required to compute metadata: {e}")
        return None, None, None

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load config for {model_id}: {e}")
        cfg = None

    max_tokens = _get_max_tokens_from_config(cfg) if cfg is not None else None
    embed_dim = _get_embed_dim_from_config(cfg)

    n_params: Optional[int] = None
    try:
        with torch.no_grad():  # type: ignore[attr-defined]
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            try:
                model.eval()
            except Exception:
                pass
            n_params = _count_parameters_hf(model)
            # If embed_dim still unknown, try inspecting model.config
            if embed_dim is None:
                try:
                    embed_dim = _get_embed_dim_from_config(
                        getattr(model, "config", None)
                    )
                except Exception:
                    pass
            # Free memory
            try:
                del model
            except Exception:
                pass
    except Exception as e:
        print(f"Failed to load model for {model_id} to count parameters: {e}")

    return n_params, max_tokens, embed_dim


def update_model_meta(json_path: Path, model_id: Optional[str] = None) -> None:
    if not json_path.exists():
        raise FileNotFoundError(f"model_meta.json not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Determine model id
    mid = model_id or meta.get("name")
    if not mid:
        raise ValueError("Model id not provided and not found in JSON under 'name'.")

    n_params, max_tokens, embed_dim = compute_hf_meta(mid)

    if n_params is not None:
        meta["n_parameters"] = int(n_params)
    if max_tokens is not None:
        meta["max_tokens"] = int(max_tokens)
    if embed_dim is not None:
        meta["embed_dim"] = int(embed_dim)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    print(
        f"Updated {json_path} with n_parameters={meta.get('n_parameters')} max_tokens={meta.get('max_tokens')} embed_dim={meta.get('embed_dim')}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Update model_meta.json with HF-derived n_parameters, max_tokens, and embed_dim (when available)"
    )
    ap.add_argument(
        "--json", type=str, required=True, help="Path to model_meta.json to update"
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional HF model id; defaults to JSON 'name'",
    )
    args = ap.parse_args()

    update_model_meta(Path(args.json), args.model)


if __name__ == "__main__":
    main()
