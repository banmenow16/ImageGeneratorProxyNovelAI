"""Backend-neutral generation interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
	from imagine_generator_proxy.prompts import PromptMassager


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


@dataclass
class Backend:
	"""A selectable generation backend: generator + its ruleset-matched massager."""
	name: str
	display: str
	generator: "ImageGenerator"
	massager: "PromptMassager"  # noqa: F821 — TYPE_CHECKING import
