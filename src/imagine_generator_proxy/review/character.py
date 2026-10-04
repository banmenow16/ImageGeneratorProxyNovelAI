"""Persona-injected review: judges images against a character's stored
self-description, on top of the inner reviewer's normal QC."""

from __future__ import annotations

import logging
from typing import Optional

from imagine_generator_proxy.review.base import Reviewer, Verdict

LOGGER = logging.getLogger(__name__)


class CharacterReviewer:
	"""Decorator over any Reviewer: when the character has a stored
	description, the review prompt is grounded with her appearance so the
	vision model judges likeness as well as quality. Thin by design — all
	verdict logic stays in the inner reviewer."""

	def __init__(self, inner: Reviewer, name: str, description: str):
		self._inner = inner
		self._name = name
		self._description = description

	@property
	def last_error(self) -> str:
		return getattr(self._inner, "last_error", "")

	def _context_note(self) -> str:
		return (
			f"CHARACTER GROUNDING: You are judging images of {self._name}. "
			f"This is how she describes herself: \"{self._description[:1200]}\" "
			"FIRST assess LIKENESS — does the figure match this appearance? "
			"Name specific mismatches (hair, eyes, build, features) in your notes, "
			"prefixed 'LIKENESS:'. Then do your normal quality review. A likeness "
			"mismatch alone is a defect worth rejecting over only when it is clear "
			"and specific, not a stylistic difference."
		)

	async def review(self, image: bytes, prompt: str, attempt: int = 0,
			previous_notes: str = "", character: str = "",
			context_note: str = "") -> Verdict:
		note = context_note or self._context_note()
		return await self._inner.review(image, prompt, attempt=attempt,
			previous_notes=previous_notes, character=character, context_note=note)
