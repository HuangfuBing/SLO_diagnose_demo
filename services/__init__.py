"""Service layer for SPM/VLM inference wrappers.

This package exposes light-weight, side-effect-free helpers that the Gradio UI
can import without immediately loading large model weights. The actual loading
is performed lazily inside each service module.
"""

from .spm_api import SpmClient, SpmResult  # noqa: F401
from .vlm_api import VlmClient, VlmResult  # noqa: F401
from .spm_runner_impl import build_spm_runner  # noqa: F401
from .vlm_runner_impl import build_vlm_runner  # noqa: F401
from .label_map import enrich_disease_probs, load_labelmap, resolve_disease_label  # noqa: F401
