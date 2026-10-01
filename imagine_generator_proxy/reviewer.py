"""Reviewer seam for post-generation image review (Phase 2 placeholder).

A future vision-model reviewer (e.g. GLM vision over plain HTTP) implements
the Reviewer protocol and is stored on app["reviewer"]; the generation flow
consults it after image bytes arrive and before the image is cached, so a
Phase 2 retry loop can regenerate with an adjusted prompt and a recomputed
digest without restructuring the caching logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class Verdict:
	approved: bool
	score: Optional[float] = None
	notes: str = ""


class Reviewer(Protocol):
	"""Score a generated image against the prompt it was generated from."""

	async def review(self, image: bytes, prompt: str) -> Verdict: ...


class NoopReviewer:
	"""Approves everything; the default until a real reviewer exists."""

	async def review(self, image: bytes, prompt: str) -> Verdict:
		return Verdict(approved=True, notes="noop reviewer")
