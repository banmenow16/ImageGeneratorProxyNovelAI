"""Reviewer seam for post-generation image review (Phase 2 placeholder).

A future vision-model reviewer (e.g. GLM vision over plain HTTP) implements
the Reviewer protocol and is stored on app["reviewer"]; the generation flow
consults it after image bytes arrive and before the image is cached, so a
Phase 2 retry loop can regenerate with an adjusted prompt and a recomputed
digest without restructuring the caching logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass
class Verdict:
	approved: bool
	score: Optional[float] = None
	notes: str = ""
	# None on approval; "quality" (retryable, fail-open) or "minor"
	# (fail-closed: the image is never served) on rejection.
	category: Optional[str] = None
	# Reviewer-proposed repair tags for the next regeneration attempt:
	# positive-prompt additions and negative-prompt additions ('neg:' entries).
	fix_add: List[str] = field(default_factory=list)
	fix_negate: List[str] = field(default_factory=list)


class Reviewer(Protocol):
	"""Score a generated image against the prompt it was generated from.

	`attempt` (0-based) and `previous_notes` (the prior attempt's rejection
	notes) carry the regeneration session, so reviewers can verify whether
	earlier defects were fixed instead of starting from scratch.
	"""

	async def review(self, image: bytes, prompt: str, attempt: int = 0,
			previous_notes: str = "", character: str = "",
			context_note: str = "") -> Verdict: ...


class NoopReviewer(Reviewer):
	"""Approves everything; the default until a real reviewer exists."""

	async def review(self, image: bytes, prompt: str, attempt: int = 0,
			previous_notes: str = "", character: str = "",
			context_note: str = "") -> Verdict:
		return Verdict(approved=True, notes="noop reviewer")
