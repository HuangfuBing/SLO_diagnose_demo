"""
Wrapper for VLM inference with optional SPM feature injection.

Designed to stay decoupled from the underlying Swift/Qwen registration scripts.
You can plug in a callable that performs the actual model invocation. When
MOCK_VLM=1, the wrapper returns a deterministic dummy report so that the Gradio
demo remains runnable without GPU weights.
"""

from __future__ import annotations

import os
import time
import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from PIL import Image


@dataclass
class VlmResult:
    """Structured result from the VLM."""

    report: str
    raw_output: Optional[Any] = None
    debug: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


class VlmClient:
    """
    Facade for VLM generation.

    Parameters
    ----------
    runner : Callable
        Callable accepting (prompt: str, image_paths: Sequence[str], spm_feat_path: Optional[str])
        and returning VlmResult.
    workdir : str | Path
        Directory for temporary image serialization when Pillow images are passed.
    """。

    def __init__(
        self,
        runner: Optional[
            Callable[[str, Sequence[str], Optional[str]], VlmResult]
        ] = None,
        workdir: Optional[str | Path] = None,
    ):
        self.runner = runner
        self.workdir = Path(workdir or ".cache_vlm")
        self.workdir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        prompt: str,
        images: Iterable[str | Image.Image],
        spm_feat_path: Optional[str] = None,
    ) -> VlmResult:
        paths = list(self._ensure_paths(images))
        if self.runner is not None:
            return self.runner(prompt, paths, spm_feat_path)
        if os.getenv("MOCK_VLM", "0") == "1":
            return self._mock_result(prompt, paths, spm_feat_path)
        raise RuntimeError(
            "No VLM runner provided. Set MOCK_VLM=1 for mock mode or supply a runner callable."
        )

    # Helpers --------------------------------------------------------
    def _ensure_paths(self, images: Iterable[str | Image.Image]):
        for img in images:
            if isinstance(img, str):
                yield img
            elif isinstance(img, Image.Image):
                name = f"vlm_upload_{uuid.uuid4().hex}.png"
                path = self.workdir / name
                img.save(path)
                yield str(path)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

    def _mock_result(
        self, prompt: str, paths: Sequence[str], spm_feat_path: Optional[str]
    ) -> VlmResult:
        report = (
            "【Mock 报告】疑似病灶影像发现：\n"
            "- 病变范围：有限\n- 建议：随访与进一步影像检查\n"
            "提示：当前为 MOCK_VLM=1，占位输出；实际部署时请接入真实 VLM 推理。"
        )
        return VlmResult(
            report=report,
            debug={
                "mock": True,
                "prompt": prompt,
                "images": list(paths),
                "spm_feat_path": spm_feat_path,
                "timestamp": time.time(),
            },
        )


def make_default_vlm_client(
    runner: Optional[
        Callable[[str, Sequence[str], Optional[str]], VlmResult]
    ] = None,
    workdir: Optional[str | Path] = None,
) -> VlmClient:
    """
    Create a VLM client. Pass a runner that wraps the Swift/Qwen inference call.

    Example runner signature:
        def real_vlm_runner(prompt, image_paths, spm_feat_path=None) -> VlmResult:
            # ... load model once, call generate, return VlmResult(report=..., raw_output=...)
            return VlmResult(report=generated_text, raw_output=raw)
    """
    return VlmClient(runner=runner, workdir=workdir)

