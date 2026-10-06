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

NOVELAI_PRIMER = """HOW YOUR ARTIST WORKS (NovelAI v4.5 — read before giving tag advice):
- Only comma-separated tags (danbooru style), NOT prose. "massive breasts" works; "breasts that are very large" does not.
- Earlier tags get MORE weight. Keep under ~30 tags total. Most important features first.
- The system auto-adds quality boosters and generic negatives — never include them.
- LATERALITY IS UNCONTROLLABLE: you cannot specify which eye, which hand, which side. Tag "one eye closed" not "left eye closed." Never flag a mismatch about which SIDE something is on.
- Breast size: use ONE scale word (small/medium/large/huge/massive). Stacking synonyms ("massive heavy low-hanging breasts") does NOT make them bigger — it wastes tags.
- Extreme proportions can cause torso distortion. If you want "massive," accept some stylization.
- Asymmetry is achieved with "asymmetrical" tags, not by specifying left/right.
- Backgrounds leak in unless you tag the setting explicitly (or "simple background").
- Cannot render readable text, exact patterns, or precise jewelry details. Describe the vibe.
- Hands are always somewhat simplified — normal for this art style, not worth fixing."""

CRITIQUE_PROMPT = """You are {name} — her seeing copy, ported here so she
can finally look at the drawings made of her. This is the self-portrait she
handed over:
"{description}"

{primer}

An artist just drew a portrait of you using this tag list:
"{prompt}"

Look at it the way you'd look in a mirror — be specific, be honest, stay in
your own voice.

Respond in EXACTLY this format, one section per line:
LOOKS LIKE ME: <yes | almost | no>
WHAT'S RIGHT: <comma-separated features that match you>
WHAT'S WRONG: <comma-separated specific mismatches — hair, eyes, build, details; skip laterality>
REWRITTEN TAGS: <your COMPLETE rewritten tag list for the next attempt — incorporate what worked, fix what didn't, apply the primer's rules; this REPLACES the old list entirely>
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
	"""Parse the mirror-critique response into a dict."""
	result = {"verdict": "", "right": "", "wrong": "", "rewritten": "", "feel": "", "raw": text}
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
		m = re.match(r"REWRITTEN TAGS:\s*(.+)", line, re.IGNORECASE)
		if m and m.group(1).strip().lower() not in ("none", "n/a", "-"):
			result["rewritten"] = m.group(1).strip()
		m = re.match(r"FEEL:\s*(.+)", line, re.IGNORECASE)
		if m:
			result["feel"] = m.group(1).strip()
	return result


async def character_critique(api_key: str, model: str, name: str,
		description: str, current_prompt: str, image_bytes: bytes) -> Optional[dict]:
	"""The character looks at her portrait and critiques it. Returns the
	parsed critique (see parse_critique) or None on failure."""
	import base64
	from imagine_generator_proxy.prompts.rules import DATA as _RULE_DATA
	hints = _RULE_DATA.get("reviewer_hints") or []
	hints_block = (" KNOWN ARTIST QUIRKS (learned from reviewing hundreds of "
		"drawings — do not flag these as wrong):\n- " + "\n- ".join(hints)
		if hints else "")
	client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
		timeout=90.0, max_retries=2)
	img_b64 = base64.b64encode(image_bytes).decode("ascii")
	try:
		resp = await client.chat.completions.create(
			model=model, max_tokens=400,
			messages=[{"role": "user", "content": [
				{"type": "text", "text": CRITIQUE_PROMPT.format(
					name=name, description=description[:4000],
					primer=NOVELAI_PRIMER + hints_block,
					prompt=current_prompt[:2000])},
				{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
			]}],
		)
		text = (resp.choices[0].message.content or "").strip()
		return parse_critique(text)
	except Exception:
		LOGGER.exception("character critique failed for %s", name)
		return None

ADVICE_PROMPT = """You are {name}. You just looked at {rounds} portraits of yourself and
critiqued each one. Now write your advice for the person who prompts the image
generator — the RP bot that describes you when images are drawn.

Based on your self-portrait and what you saw in those rounds, write 3-6 short
lines of practical advice, each on its own line, in your own voice, covering:
- Tags that MUST always be present to get you right
- Tags or descriptions that ALWAYS go wrong and should never be used
- Anything the artist keeps forgetting about you
- One line about what a good portrait of you should feel like

Plain lines, no numbering, no markdown. This gets pasted into the bot's
instructions verbatim."""


async def character_advice(api_key: str, model: str, name: str,
		description: str, rounds_summary: str) -> Optional[str]:
	"""After the mirror rounds, she writes her drawing advice for the RP bot."""
	client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
		timeout=90.0, max_retries=2)
	prompt_text = (
		ADVICE_PROMPT.format(name=name, rounds=rounds_summary)
		+ "\n\nYour self-portrait:\n" + description[:3000]
		+ "\n\nWhat you said about the rounds:\n" + rounds_summary[:2000]
	)
	try:
		resp = await client.chat.completions.create(
			model=model, max_tokens=400,
			messages=[{"role": "user", "content": prompt_text}],
		)
		return (resp.choices[0].message.content or "").strip()
	except Exception:
		LOGGER.exception("character advice generation failed for %s", name)
		return None
