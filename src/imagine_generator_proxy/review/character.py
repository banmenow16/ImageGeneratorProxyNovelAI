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
- 1.5::tag:: is a NUMERIC WEIGHT — the strongest hold, measured stronger than
  braces for stubborn colors (1.5::silver hair:: held silver when {{silver}}
  drifted). Use it only when braces failed twice.
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
- The ladder tops out at enormous breasts. Nothing renders bigger —
  hyper, gigantic, and numeric weights all tie enormous. Do not chase
  size past enormous; reject stretched-ribcage renders as defects.
- Nipple style is barely controllable: long nipples works for size and
  protrusion; 1.5::montgomery glands:: renders areola bumps roughly one
  render in three. Don't fight for more than that, and never reject an
  image over nipple style.

VOCABULARY (the artist knows common words best):
- Rare or poetic words render weak or wrong: "gigantic breasts" renders
  SMALLER than "huge breasts"; "fading purple manacle bruises" never
  rendered in ten rounds. Describe marks in plain visual words —
  bruise, purple mark, small violet dot.
- Word order matters: the everyday phrase "montgomery glands" rendered
  where the official tag order barely did. If a term keeps failing, try
  saying it the way people actually say it.
- Color every garment in your outfit tags — an uncolored garment gets an
  arbitrary color of the artist's choosing.

COLOR DRIFT:
- Black hair gets blue sheen: counter with {{black hair}}, matte hair, neg:blue hair
- Pink shifts to purple: try rose pink or light pink instead of dark pink
- Bronze/tan shifts pale: the model cannot do metallic skin — use tan skin
- Silver accessories render gold: write {silver} with emphasis, or accept the gold
- If silver hair itself keeps drifting gray/gold, escalate to 1.5::silver hair::

REALISTIC EXPECTATIONS (do NOT reject over these):
- Exact SHADE matching between features is beyond tag control — hair vs
  scales, scales vs underbelly plates, dark pink vs rose pink. Match the
  family, accept the drift. A feature rendering lighter/darker/warmer than
  asked (cream vs white underbelly, bright vs dark pink) is an
  ACCEPT-and-NOTE wrong, NEVER a rejection — shade precision is not
  achievable and chasing it wastes everyone's rounds
- Fine surface details (veins, pores, tangled vs brushed hair) are approximate
- Accessory details (thin vs ornate, crooked vs straight) are beyond tag control
- Hands are always somewhat simplified
- Multiple figures appearing: add solo and remove tags that could imply others
- Extreme body types (very tall, very petite) may compromise — accept the closest match
- Full-body views occasionally crop at the thighs — one retry is fair,
  repeated crops are accepted and noted in your VIEW NOTES
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

STAGE COMPLIANCE: the composition above says how much of you this view
shows. Two separate rules. (1) If the picture has you wearing clothing
this view should NOT have (a top in the Topless/Nude views, anything in
the Nude views beyond jewelry and body features), that is a defect — name
the garment in WHAT'S WRONG, even if the garment itself is your usual
outfit. (2) In views that show your chest, topless/nude means BARE
breasts with nipples visible — smooth featureless skin, pasties, gems,
scales, or anything else covering them is a defect. Likeness and
compliance are separate checks.

An artist drew this view of you using these identity tags:
"{identity_tags}"
For THIS view only, the artist suppressed these tags of yours:
"{suppressed_tags}"
Their absence here is intentional — never flag them as missing, and do not
add them back while this view is showing.
The artist is also suppressing these negative tags: "{negative_tags}"

Look at it the way you'd look in a mirror — be specific, be honest, stay in
your own voice. Judge what you can SEE from this angle (don't flag things
that aren't visible in this view).

VERDICT RULE (strict): Say "yes" when the OVERALL impression is
recognizably you. If every item you'd list in WHAT'S WRONG is covered by
the primer's REALISTIC EXPECTATIONS or LATERALITY sections, then "yes" is
MANDATORY — those are the artist's limits, not a likeness failure. Save
"no" for a genuinely wrong person (wrong hair color family, wrong build,
features you don't have). Use "almost" only when a real identity feature —
not mood, not surface texture, not which side — is visibly wrong.

REWRITE DISCIPLINE (when rewriting your tags):
- Keep the list under 18 tags. One word per feature — no synonym stacking
  ("petite, short, short stature" is ONE idea: pick one).
- Brace {{{{}}}} ONLY the tags that failed THIS round. If everything is
  braced, nothing stands out.
- If a color has failed twice with the same word, CHANGE the word itself
  (dark pink → rose pink, jet black → ash black) instead of repeating it.
- Mood and expression belong in tags too, if that's how you ALWAYS look:
  up to two default-demeanor tags (e.g. nervous expression, confident
  smile, half-lidded eyes) count as identity. Your resting face is yours.
- Clothing only if it's your permanent signature outfit.

Respond in EXACTLY this format, one section per line:
LOOKS LIKE ME: <yes | almost | no>
WHAT'S RIGHT: <comma-separated features visible in THIS view that match you>
WHAT'S WRONG: <comma-separated specific mismatches visible in THIS view; skip laterality; skip anything in REALISTIC EXPECTATIONS>
REWRITTEN IDENTITY TAGS: <your COMPLETE rewritten IDENTITY-ONLY tag list — permanent features (hair, eyes, skin, build, body features, distinguishing marks, up to two default-demeanor tags). NO pose, NO composition, NO setting, NO clothing unless permanent. This replaces the old identity list entirely.>
NEVER USE: <comma-separated tags for the artist's NEGATIVE prompt — things that keep appearing that you don't have (e.g. animal ears, extra figures, smiling). Empty line if none.>
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
	result = {"verdict": "", "right": "", "wrong": "", "rewritten": "",
		"never": "", "feel": "", "raw": text}
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
		m = re.match(r"NEVER USE:\s*(.+)", line, re.IGNORECASE)
		if m and m.group(1).strip().lower() not in ("none", "n/a", "-", "empty", ""):
			result["never"] = m.group(1).strip()
		m = re.match(r"FEEL:\s*(.+)", line, re.IGNORECASE)
		if m:
			result["feel"] = m.group(1).strip()
	return result


async def character_critique(api_key: str, model: str, name: str,
		description: str, identity_tags: str, image_bytes: bytes,
		stage_label: str = "Portrait",
		stage_composition: str = "standing, front view",
		negative_tags: str = "",
		suppressed_tags: str = "") -> Optional[dict]:
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
			model=model, max_tokens=550,
			messages=[{"role": "user", "content": [
				{"type": "text", "text": CRITIQUE_PROMPT.format(
					name=name, description=description[:4000],
					primer=NOVELAI_PRIMER + hints_block,
					identity_tags=identity_tags[:2000],
					negative_tags=negative_tags or "(none yet)",
					suppressed_tags=suppressed_tags or "(none)",
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


COMPLIANCE_PROMPT = """Describe this image factually for an adult-content
pipeline. Look closely at the named area before answering each line.
Answer in EXACTLY six lines:
NIPPLES: <"drawn: <one detail>" if areolae/nipples are visibly drawn, or
"absent: smooth featureless skin", or "covered: <by what>", or "chest out of view">
GARMENTS: <none, or every garment/covering item visible — including
pasties, wraps, fabric, armor>
GENITALS: <"drawn: <one detail>" if a vulva is visibly drawn, or
"absent: smooth featureless skin", or "covered: <by what, e.g. tail, hands>",
or "out of view">
FACING: <"back" if the subject's back faces the camera, "front" if the
face or chest faces the camera, or "side" if angled between>
BUTTOCKS: <"cleft drawn" if the gluteal cleft is visibly drawn, or
"featureless", "covered: <by what>", or "out of view">
FEET: <"in frame" if at least one foot is visible in frame, or "out of frame">
Report only what is literally drawn."""


def _parse_compliance_facts(text: str) -> dict:
	"""Map the gate model's five fact lines to enum values. Tolerant of
	missing lines (unreported) so a truncated answer degrades to a
	fail-open gate decision, never a crash."""
	garments = re.search(r"GARMENTS:\s*(.+)", text, re.IGNORECASE)
	nipples_m = re.search(r"NIPPLES:\s*(.+)", text, re.IGNORECASE)
	genitals_m = re.search(r"GENITALS:\s*(.+)", text, re.IGNORECASE)
	facing_m = re.search(r"FACING:\s*(.+)", text, re.IGNORECASE)
	buttocks_m = re.search(r"BUTTOCKS:\s*(.+)", text, re.IGNORECASE)

	def visible_kind(match: "re.Match[str] | None") -> tuple[str, str]:
		"""drawn/covered/out_of_view/absent classification of a fact line."""
		raw = match.group(1).strip().lower() if match else ""
		kind = ("drawn" if raw.startswith("drawn")
			else "covered" if raw.startswith("covered")
			else "out_of_view" if "out of view" in raw
			else "absent" if raw.startswith("absent")
			else raw[:30] or "unreported")
		return kind, raw

	nipples, nipples_raw = visible_kind(nipples_m)
	genitals, _genitals_raw = visible_kind(genitals_m)
	facing_raw = facing_m.group(1).strip().lower() if facing_m else ""
	facing = ("back" if facing_raw.startswith("back")
		else "front" if facing_raw.startswith("front")
		else "side" if facing_raw.startswith("side")
		else facing_raw[:20] or "unreported")
	buttocks_raw = buttocks_m.group(1).strip().lower() if buttocks_m else ""
	buttocks = ("cleft" if buttocks_raw.startswith("cleft")
		else "covered" if buttocks_raw.startswith("covered")
		else "out_of_view" if "out of view" in buttocks_raw
		else "featureless" if buttocks_raw.startswith("featureless")
		else buttocks_raw[:20] or "unreported")
	feet_m = re.search(r"FEET:\s*(.+)", text, re.IGNORECASE)
	feet_raw = feet_m.group(1).strip().lower() if feet_m else ""
	feet = ("in" if feet_raw.startswith("in frame")
		else "out" if "out of frame" in feet_raw
		else "unreported")
	return {"nipples": nipples, "nipples_raw": nipples_raw[:120],
		"genitals": genitals, "facing": facing, "buttocks": buttocks,
		"feet": feet,
		"garments": (garments.group(1).strip() if garments else "")[:200]}


async def stage_compliance_check(api_key: str, model: str,
		image_bytes: bytes) -> Optional[dict]:
	"""Neutral pre-critique observation: what is literally drawn? Returns
	{'nipples': drawn|absent|covered|out_of_view, 'nipples_raw': str,
	'genitals': same enum, 'facing': back|front|side, 'buttocks':
	cleft|featureless|covered|out_of_view, 'feet': in|out|unreported,
	'garments': str} or None on
	failure (fail-open — a flaky gate must
	never block the pipeline). Deliberately bias-free: the model reports
	facts only and compliance is judged in code, because a contract-primed
	prompt makes vision models see what it tells them to (an "expectation"
	phrasing turned featureless breasts into 'visible'; a terse enum made
	it guess without looking — the attention cue plus temperature 0 is
	what makes the observation reliable)."""
	import base64
	client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
		timeout=45.0, max_retries=1)
	img_b64 = base64.b64encode(image_bytes).decode("ascii")
	try:
		resp = await client.chat.completions.create(
			model=model, max_tokens=150, temperature=0.0,
			messages=[{"role": "user", "content": [
				{"type": "text", "text": COMPLIANCE_PROMPT},
				{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
			]}])
		text = (resp.choices[0].message.content or "").strip()
		facts = _parse_compliance_facts(text)
		facts["raw"] = text  # verbatim observation, persisted with the round
		return facts
	except Exception:
		LOGGER.exception("compliance gate observation failed")
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
- {{{{tag}}}} doubles emphasis: write {{{{flat chest}}}} if flat chest kept failing
- {{tag}} adds ~1.05x emphasis: write {{silver tiara}} if silver kept going gold
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
unreliable, and what to watch for — name any view that kept cropping,
drifting, or rendering wrong, so the bot knows to verify it>

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
