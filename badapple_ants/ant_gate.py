"""Zero-shot CLIP ant-presence gate to filter out empty crops, glare, bokeh, and labels."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .download import _thumb_path
from .geometry import render_crop

DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b79k"
MIN_ANT_SCORE = 0.50

ANT_PROMPTS = (
    "a close-up photograph of an ant",
    "a macro photo of an ant insect",
    "a photo of an ant with legs and antennae",
    "a pinned ant museum specimen",
)

NON_ANT_PROMPTS = (
    "a close-up of an empty wall, floor, wood, or painted surface",
    "a dark shadow or geometric edge with no insect",
    "bright camera flash glare or reflection on plastic or glass",
    "a blurry out-of-focus background bokeh",
    "a human finger or hand",
    "a printed paper text label",
)


def ant_probs_from_logits(logits: torch.Tensor, n_pos: int) -> torch.Tensor:
    """Compute probability in [0, 1] that logits match one of the first `n_pos` ant prompts."""
    probs = torch.softmax(logits, dim=-1)
    return probs[..., :n_pos].sum(dim=-1)


def _resolve_thumb_path(item: dict, cache_dir: Path) -> Path | None:
    raw = item.get("thumb_path")
    if isinstance(raw, str) and raw:
        for candidate in (Path(raw), Path(raw.replace("\\", "/"))):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    fallback = _thumb_path(cache_dir, item)
    if fallback.is_file() and fallback.stat().st_size > 0:
        return fallback
    return None


def _pose_cache_key(cand: dict) -> tuple[int, int, int, int, int, int]:
    return (
        int(cand["image_idx"]),
        int(round(float(cand["cx_frac"]) * 1000.0)),
        int(round(float(cand["cy_frac"]) * 1000.0)),
        int(round(float(cand["w_frac"]) * 1000.0)),
        int(round(float(cand["theta"]) * 1000.0)),
        int(bool(cand["mirrored"])),
    )


class AntGate:
    """Batched GPU CLIP classifier checking that a visible ant appears in candidate crops."""

    def __init__(
        self,
        device: torch.device,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        batch_size: int = 128,
        max_open_images: int = 512,
    ) -> None:
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_open_images = max(1, int(max_open_images))

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess

        tokenizer = open_clip.get_tokenizer(model_name)
        all_prompts = list(ANT_PROMPTS) + list(NON_ANT_PROMPTS)
        self.n_pos = len(ANT_PROMPTS)
        tokens = tokenizer(all_prompts).to(device)

        with torch.no_grad():
            text_feat = self.model.encode_text(tokens).float()
            self.text_features = F.normalize(text_feat, dim=-1)
            self.logit_scale = float(self.model.logit_scale.exp().clamp(max=100.0).item())

        self._cache: dict[tuple[int, int, int, int, int, int], float] = {}
        self._img_cache: OrderedDict[Path, Image.Image] = OrderedDict()

    def _load_rgb(self, path: Path) -> Image.Image | None:
        cached = self._img_cache.get(path)
        if cached is not None:
            self._img_cache.move_to_end(path)
            return cached
        try:
            with Image.open(path) as raw:
                loaded = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception:
            return None
        self._img_cache[path] = loaded
        if len(self._img_cache) > self.max_open_images:
            self._img_cache.popitem(last=False)
        return loaded

    def score_candidates(
        self,
        candidates: list[dict],
        items: list[dict],
        cache_dir: Path,
    ) -> list[float]:
        """Return CLIP ant-presence probability [0, 1] for each candidate dict."""
        if not candidates:
            return []

        keys = [_pose_cache_key(c) for c in candidates]
        pending_keys: list[tuple[int, int, int, int, int, int]] = []
        pending_tensors: list[torch.Tensor] = []
        seen_pending: set[tuple[int, int, int, int, int, int]] = set()

        for cand, key in zip(candidates, keys):
            if key in self._cache or key in seen_pending:
                continue
            seen_pending.add(key)

            item = items[int(cand["image_idx"])]
            thumb_path = _resolve_thumb_path(item, cache_dir)
            if thumb_path is None:
                self._cache[key] = 0.0
                continue
            pil_img = self._load_rgb(thumb_path)
            if pil_img is None:
                self._cache[key] = 0.0
                continue
            crop = render_crop(
                pil_img,
                224,
                168,
                float(cand["cx_frac"]),
                float(cand["cy_frac"]),
                float(cand["w_frac"]),
                float(cand["theta"]),
                bool(cand["mirrored"]),
            )
            pending_keys.append(key)
            pending_tensors.append(self.preprocess(crop))

        for start in range(0, len(pending_tensors), self.batch_size):
            batch = torch.stack(pending_tensors[start : start + self.batch_size]).to(self.device)
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        img_feat = self.model.encode_image(batch)
                else:
                    img_feat = self.model.encode_image(batch)
                img_feat = F.normalize(img_feat.float(), dim=-1)
                logits = self.logit_scale * (img_feat @ self.text_features.T)
                probs = ant_probs_from_logits(logits, self.n_pos).detach().cpu().tolist()
            for key, prob in zip(pending_keys[start : start + self.batch_size], probs):
                self._cache[key] = float(prob)

        return [self._cache[key] for key in keys]
