"""Backend-neutral generation interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class GeneratedImage:
	"""A single image returned by a generation backend."""

	name: str
	data: bytes
	seed: Optional[int] = None


class ImageGenerator(Protocol):
	"""Backend-neutral image generation interface.

	Implementations (NovelAI today; e.g. Civitai/local model servers later)
	generate one image for a prompt/negative-prompt pair. `prompt_is_complete`
	is True when the caller (the prompt massager) already assembled the full
	quality-tagged prompt: backends that auto-decorate prompts must not add
	their own tags then. Backends with no auto-decoration may ignore it.
	Returns None when the backend stays busy/unavailable after its retries;
	raise on hard errors.
	"""

	async def generate(self, prompt: str, negative_prompt: str,
			prompt_is_complete: bool) -> Optional[GeneratedImage]: ...
