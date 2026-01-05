"""
Lightweight wrapper for SPM-based upstream model inference.

The goal is to provide a clean Python interface that the Gradio UI can call
without modifying the original training / evaluation scripts. The wrapper
supports two modes:

1) Production mode: plug in a user-provided callable that actually runs the
   model (e.g., invoking `main_calib_eval.py` or a packaged checkpoint).
2) Mock mode: controlled by the environment variable MOCK_SPM=1, returns
   deterministic dummy outputs so the Gradio UI remains runnable even when
   model weights are unavailable on the current machine.
"""

from __future__ import annotations

import os
import time
import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from PIL import Image


# ----------------------------
# Data containers
# ----------------------------
@dataclass
class SpmResult:
    """Structured result produced by the SPM client."""

    lesion_probs: List[Dict[str, Any]]
    disease_probs: List[Dict[str, Any]]
    thresholds: Dict[str, Any]
    spm_feat_path: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


class SpmClient:
    """
    Simple client facade.

    Parameters
    ----------
    runner : Callable
        A callable that takes a list of image paths and returns `SpmResult`.
        If None, falls back to mock mode when MOCK_SPM=1, otherwise raises.
    workdir : str | Path
        Location to store intermediate outputs such as serialized features.
    """

    def __init__(
        self,
        runner: Optional[Callable[[Sequence[str]], SpmResult]] = None,
        workdir: Optional[str | Path] = None,
    ):
        self.runner = runner
        self.workdir = Path(workdir or ".cache_spm")
        self.workdir.mkdir(parents=True, exist_ok=True)

    # Public API -----------------------------------------------------
    def __call__(self, images: Iterable[str | Image.Image]) -> SpmResult:
        paths = list(self._ensure_paths(images))
        if self.runner is not None:
            return self.runner(paths)
        if os.getenv("MOCK_SPM", "0") == "1":
            return self._mock_result(paths)
        raise RuntimeError(
            "No SPM runner provided. Set MOCK_SPM=1 for mock mode or supply a runner callable."
        )

    # Helpers --------------------------------------------------------
    def _ensure_paths(self, images: Iterable[str | Image.Image]) -> Iterable[str]:
        for img in images:
            if isinstance(img, str):
                yield img
            elif isinstance(img, Image.Image):
                name = f"upload_{uuid.uuid4().hex}.png"
                path = self.workdir / name
                img.save(path)
                yield str(path)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

    def _mock_result(self, paths: Sequence[str]) -> SpmResult:
        # deterministic mock probabilities
        lesion_probs = [
            {"id": f"lesion_{i}", "prob": round(0.3 + 0.05 * i, 3)} for i in range(5)
        ]
        disease_probs = [
            {"id": "disease_A", "prob": 0.71, "threshold": 0.5},
            {"id": "disease_B", "prob": 0.42, "threshold": 0.5},
        ]
        thresholds = {"default": 0.5}
        # fake spm feature file
        feat_path = self.workdir / f"spm_feat_{int(time.time())}.npy"
        feat_path.write_bytes(b"")  # placeholder; real runner should save numpy array
        return SpmResult(
            lesion_probs=lesion_probs,
            disease_probs=disease_probs,
            thresholds=thresholds,
            spm_feat_path=str(feat_path),
            debug={"mock": True, "inputs": list(paths)},
        )


# Convenience factory -----------------------------------------------
def make_default_spm_client(
    runner: Optional[Callable[[Sequence[str]], SpmResult]] = None,
    workdir: Optional[str | Path] = None,
) -> SpmClient:
    """
    Create a default SPM client. The user can pass an actual runner callable that
    wraps `main_calib_eval.py` or any custom inference code. Example signature:

        def real_runner(paths: Sequence[str]) -> SpmResult:
            # ... load model once, run inference, save npy features, etc.
            return SpmResult(...)

    """
    return SpmClient(runner=runner, workdir=workdir)

