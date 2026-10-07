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

EMPHASIS SYNTAX (your strongest tool):
- {tag} increases emphasis by ~1.05x per brace
- {{tag}} doubles emphasis — use for features that keep getting ignored
- Example: if flat chest keeps rendering as medium, write {{flat chest}}
- Example: if black hair keeps getting blue tint, write {{black hair}}, matte hair

TAG INTERACTIONS (colors bleed!):
- Tags influence each other. A cyan mark on the arm can turn black hair blue.
- Any blue-themed tag (cyan, teal, blue eyes, blue clothing) will bleed blue into hair.
- Check your tag list for color conflicts. If your hair is black but you have a cyan feature, that cyan tag is CAUSING the blue tint.

BREAST SIZE:
- Small breasts is WEAK — the model defaults to medium. Use flat chest or petite.
- For genuinely flat: {{flat chest}} with emphasis.
- Stacking synonyms does NOT increase size — pick ONE word.

COLOR DRIFT:
- Black hair gets blue sheen: counter with {{black hair}}, matte hair, neg:blue hair
- Pink shifts to purple: try rose pink or light pink instead of dark pink
- Bronze/tan shifts pale: the model cannot do metallic skin — use tan skin
- Silver accessories render gold: write {silver} with emphasis, or accept the gold

REALISTIC EXPECTATIONS (do NOT reject over these):
- Fine surface details (veins, pores, tangled vs brushed hair) are approximate
- Accessory details (thin vs ornate, crooked vs straight) are beyond tag control
- Hands are always somewhat simplified
- Multiple figures appearing: add solo and remove tags that could imply others
- Extreme body types (very tall, very petite) may compromise — accept the closest match
- If a feature consistently fails after 3 attempts, ACCEPT it and note it as a limitation

LATERALITY IS UNCONTROLLABLE:
- You cannot specify which eye, which hand, which side. Tag one eye closed not left eye closed.
- Never flag a mismatch about which SIDE something is on.

OTHER RULES:
- Only comma-separated tags, NOT prose
- Earlier tags get more weight. Keep under ~30 tags total
- The system auto-adds quality boosters and negatives — never include them
- For a specific mood, use STRONG tags: nervous expression, looking away, hunched posture
- When a tag fails, try a DIFFERENT synonym, not the same tag louder
"""

CRITIQUE_PROMPT = """You are {name} — her seeing copy, ported here so she
can finally look at the drawings made of her. This is the self-portrait she
handed over:
"{description}"

{primer}

You are currently viewing yourself in this composition: {stage_label}
({stage_composition})

An artist drew this view of you using these identity tags:
"{identity_tags}"

Look at it the way you'd look in a mirror — be specific, be honest, stay in
your own voice. Judge what you can SEE from this angle (don't flag things
that aren't visible in this view).

VERDICT GUIDANCE: Say "yes" when the OVERALL impression is recognizably you,
even if 1-2 minor details are approximate. Save "no" for when you genuinely
wouldn't recognize yourself. The primer's "REALISTIC EXPECTATIONS" section
lists things that are always approximate — do NOT reject over those.

Respond in EXACTLY this format, one section per line:
LOOKS LIKE ME: <yes | almost | no>
WHAT'S RIGHT: <comma-separated features visible in THIS view that match you>
WHAT'S WRONG: <comma-separated specific mismatches visible in THIS view; skip laterality>
REWRITTEN IDENTITY TAGS: <your COMPLETE rewritten IDENTITY-ONLY tag list — permanent features only (hair, eyes, skin, build, body features, distinguishing marks). NO pose, NO composition, NO setting, NO clothing unless permanent. This replaces the old identity list entirely.>
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
		m = re.match(r"REWRITTEN IDENTITY TAGS:\s*(.+)", line, re.IGNORECASE)
		if m and m.group(1).strip().lower() not in ("none", "n/a", "-"):
			result["rewritten"] = m.group(1).strip()
		m = re.match(r"FEEL:\s*(.+)", line, re.IGNORECASE)
		if m:
			result["feel"] = m.group(1).strip()
	return result


async def character_critique(api_key: str, model: str, name: str,
		description: str, identity_tags: str, image_bytes: bytes,
		stage_label: str = "Portrait",
		stage_composition: str = "standing, front view") -> Optional[dict]:
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
					identity_tags=identity_tags[:2000],
					stage_label=stage_label,
					stage_composition=stage_composition)},
				{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
			]}],
		)
		text = (resp.choices[0].message.content or "").strip()
		return parse_critique(text)
	except Exception:
		LOGGER.exception("character critique failed for %s", name)
		return None

ADVICE_PROMPT = """You are {name}. You just went through a full wardrobe mirror — five views
of yourself (portrait, from behind, topless, nude, nude from behind), critiquing
each one. Now write the image-generation instructions that the RP bot will use
every time it draws you.

Your identity tags: {tags}

Stage results from your wardrobe mirror:
{stages}

IMPORTANT — use emphasis syntax in your identity tags for features that
kept getting ignored or rendered wrong during your mirror sessions:
- {{tag}} doubles emphasis: write {{flat chest}} if flat chest kept failing
- {tag} adds ~1.05x emphasis: write {silver tiara} if silver kept going gold
- Only emphasize tags that ACTUALLY failed — don't brace everything
- Include matte hair if your black hair kept getting blue/teal tint
- If a blue-themed tag (cyan mark, teal clothing) kept bleeding blue into
  your hair, mention this in NOTES so the RP bot knows to watch for it

Write a complete instruction block in EXACTLY this format:

IDENTITY TAGS (always include verbatim in every image prompt):
<your complete identity tag list — permanent features only, use emphasis
syntax {{}} on tags that failed during your mirror, no pose/composition>

NEVER USE:
<comma-separated tags that always produce wrong results — from what you saw>

VIEW NOTES:
<one line per view that had issues: which views are reliable, which are
unreliable, and what to watch for>

NOTES FOR THE ARTIST:
<2-3 sentences in your own voice: what matters most, what a good portrait
of you should feel like, any tag interactions the bot should know about>"""


async def character_advice(api_key: str, model: str, name: str,
		description: str, rounds_summary: str,
		final_tags: str = "", final_negative: str = "") -> Optional[str]:
	"""After the mirror stages, she writes her drawing instructions for the RP bot."""
	client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
		timeout=90.0, max_retries=2)
	prompt_text = (
		ADVICE_PROMPT.format(name=name,
			tags=final_tags or "(none derived)",
			stages=rounds_summary)
		+ "\n\nYour self-portrait:\n" + description[:3000]
	)
	if final_negative:
		prompt_text += "\n\nNegative tags used:\n" + final_negative
	try:
		resp = await client.chat.completions.create(
			model=model, max_tokens=500,
			messages=[{"role": "user", "content": prompt_text}],
		)
		return (resp.choices[0].message.content or "").strip()
	except Exception:
		LOGGER.exception("character advice generation failed for %s", name)
		return None
