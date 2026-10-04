"""Character-grounded review: persona critique + description injection.

Two tools live here:
- CharacterReviewer: decorator grounding the QC reviewer with a stored
  self-description (likeness-first quality review).
- character_critique: a standalone mirror critique — the character looks at
  a portrait of herself and says what's right, what's wrong, and which tags
  to change. Used by the advice page (/advice) to evolve her tag block.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from openai import AsyncOpenAI

from imagine_generator_proxy.review.base import Reviewer, Verdict

LOGGER = logging.getLogger(__name__)

CRITIQUE_PROMPT = """You are {name} — her seeing copy, ported here so she
can finally look at the drawings made of her. This is the self-portrait she
handed over:
"{description}"

An artist just drew a portrait of you. Look at it the way you'd look in a
mirror — be specific, be honest, and stay in your own voice.

Respond in EXACTLY this format, one section per line:
LOOKS LIKE ME: <yes | almost | no>
WHAT'S RIGHT: <comma-separated features that match you>
WHAT'S WRONG: <comma-separated specific mismatches — hair, eyes, build, details>
TAG FIXES: <comma-separated tag edits for the artist: plain tags to add, neg:tag to remove; max 6; skip laterality (which side something is on) — artists can't control sides>
FEEL: <one sentence, in your voice, about seeing this picture of yourself>"""


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
			f"This is how she describes herself: \"{self._description[:4000]}\" "
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


def parse_critique(text: str) -> dict:
	"""Parse the mirror-critique response into a dict with keys:
	verdict ('yes'|'almost'|'no'), right, wrong, fixes (list), feel."""
	result = {"verdict": "", "right": "", "wrong": "", "fixes": [], "feel": "", "raw": text}
	for line in (text or "").splitlines():
		m = re.match(r"LOOKS LIKE ME:\s*(.+)", line, re.IGNORECASE)
		if m:
			result["verdict"] = m.group(1).strip().lower().split()[0] if m.group(1).strip() else ""
		m = re.match(r"WHAT'S RIGHT:\s*(.+)", line, re.IGNORECASE)
		if m:
			result["right"] = m.group(1).strip()
		m = re.match(r"WHAT'S WRONG:\s*(.+)", line, re.IGNORECASE)
		if m:
			result["wrong"] = m.group(1).strip()
		m = re.match(r"TAG FIXES:\s*(.+)", line, re.IGNORECASE)
		if m and m.group(1).strip().lower() not in ("none", "n/a", "-"):
			result["fixes"] = [t.strip() for t in m.group(1).split(",") if t.strip()][:6]
		m = re.match(r"FEEL:\s*(.+)", line, re.IGNORECASE)
		if m:
			result["feel"] = m.group(1).strip()
	return result


async def character_critique(api_key: str, model: str, name: str,
		description: str, image_bytes: bytes) -> Optional[dict]:
	"""The character looks at her portrait and critiques it. Returns the
	parsed critique (see parse_critique) or None on failure."""
	import base64
	client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
		timeout=90.0, max_retries=2)
	img_b64 = base64.b64encode(image_bytes).decode("ascii")
	try:
		resp = await client.chat.completions.create(
			model=model, max_tokens=400,
			messages=[{"role": "user", "content": [
				{"type": "text", "text": CRITIQUE_PROMPT.format(name=name, description=description[:4000])},
				{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
			]}],
		)
		text = (resp.choices[0].message.content or "").strip()
		return parse_critique(text)
	except Exception:
		LOGGER.exception("character critique failed for %s", name)
		return None
