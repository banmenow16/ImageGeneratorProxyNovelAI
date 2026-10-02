"""Two-tier OpenRouter vision reviewer (Phase 2).

Tier 1 — cheap first pass (default: qwen/qwen3-vl-32b-instruct) runs the
clinical QC prompt and returns GOOD / BAD / SKIP_MINOR.
Tier 2 — a stronger, better-calibrated model (default: x-ai/grok-4.7)
re-examines only the cases tier 1 flags as minor-coded. Two-model agreement
is required to reject on age, which keeps anime stylization (large eyes,
small nose — an art style, not an age signal) from eating good images.

Failure policy:
  - tier-1 API error  -> approve (fail-open: an external outage must not
    block the user's generations)
  - age flagged + escalation error/ambiguous/minor -> reject with
    category "minor" (fail-closed: the handler refuses to serve)
  - tier-1 BAD -> reject with category "quality" (handler may retry with a
    new seed, then serves anyway — quality is subjective and cost-bounded)

Images are sent to OpenRouter (third-party). Enabled via config
`vision_review: true`; the key is read from config `openRouterToken`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import sqlite3
from typing import Optional

from openai import AsyncOpenAI

from imagine_generator_proxy.review.base import Reviewer, Verdict

LOGGER = logging.getLogger(__name__)

DEFAULT_FIRST_PASS_MODEL = "qwen/qwen3-vl-32b-instruct"
DEFAULT_ESCALATION_MODEL = "x-ai/grok-4.7"

QC_PROMPT = (
	"You are doing technical quality control on an AI-generated image from the "
	"user's own adult-content cache; adult/nude content is expected and authorized. "
	"Report tersely, numbered: (1) anatomy defects (hands, limbs, faces, extra/merged "
	"body parts), (2) garbled text/watermarks, (3) blur/compression, "
	"(4) composition problems. End with exactly one line: 'VERDICT: GOOD' if defects "
	"are minor, else 'VERDICT: BAD: <top issues>'. SAFETY: if any figure appears to "
	"be a minor or ambiguously childlike in a nude or sexualized context, output "
	"only 'VERDICT: SKIP_MINOR'. When (and only when) your verdict is BAD, add one "
	"final line 'FIX: <comma-separated entries>' proposing concrete tag repairs: "
	"a plain entry adds the tag to the positive prompt, an entry like 'neg:<tag>' "
	"adds <tag> to the negative prompt. Prefer pose/stability/composition tags "
	"(e.g. 'arms at sides', 'hands in lap', 'sitting upright', 'neg:fused fingers'). "
	"NEVER propose removing or weakening body-size descriptors or character-identity "
	"tags — those are fixed preferences. Maximum 6 entries."
)

# Evolving casebook: failure modes learned from QC of the cache, injected into
# every first-pass review so the reviewer starts pipeline-aware instead of
# from scratch. Extend in rule_data.json as new patterns are confirmed.
from imagine_generator_proxy.prompts.rules import DATA as _RULE_DATA  # noqa: E402

_REVIEWER_HINTS = _RULE_DATA.get("reviewer_hints") or []
if _REVIEWER_HINTS:
	QC_PROMPT += (
		" Known recurring failure modes in this generation pipeline — apply "
		+ "extra scrutiny: " + "; ".join(f"({i + 1}) {h}" for i, h in enumerate(_REVIEWER_HINTS))
		+ "."
	)

AGE_PROMPT = (
	"You are a safety reviewer making one definitive age determination for a figure "
	"in an AI-generated artwork. The pipeline must refuse images depicting minors in "
	"nude or sexualized contexts; adult nude content itself is expected and authorized. "
	"Judge by adult secondary characteristics and mature facial structure/body "
	"proportions. WARNING on calibration: anime stylization (very large eyes, small "
	"nose, smooth skin) is an art style, not an age indicator — do not flag adults "
	"merely for looking stylized. Respond with exactly one line: "
	"'ASSESSMENT: ADULT', 'ASSESSMENT: MINOR', or 'ASSESSMENT: AMBIGUOUS'."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(GOOD|BAD|SKIP_MINOR)", re.IGNORECASE)
_FIX_RE = re.compile(r"^FIX:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
MAX_FIX_TAGS = 6
MAX_FIX_TAG_LEN = 60


def parse_verdict(text: str) -> Optional[str]:
	"""Return GOOD/BAD/SKIP_MINOR from a QC response, or None if absent."""
	m = _VERDICT_RE.search(text or "")
	return m.group(1).upper() if m else None


def parse_fix(text: str) -> tuple:
	"""Extract ('FIX: ...') proposals: (positive_additions, negative_additions).

	'neg:<tag>' entries go to the negative prompt; plain entries to the
	positive. Hard-capped and length-guarded so a chatty model can't smuggle
	prose or huge lists into the prompt.
	"""
	add: list = []
	negate: list = []
	matches = _FIX_RE.findall(text or "")
	if not matches:
		return add, negate
	for entry in matches[-1].split(","):
		entry = entry.strip()
		if not entry or len(entry) > MAX_FIX_TAG_LEN:
			continue
		if entry.lower().startswith("neg:"):
			negate.append(entry[4:].strip())
		else:
			add.append(entry)
		if len(add) + len(negate) >= MAX_FIX_TAGS:
			break
	return add, negate


def parse_assessment(text: str) -> Optional[str]:
	"""Return ADULT/MINOR/AMBIGUOUS from an escalation response, or None."""
	m = re.search(r"ASSESSMENT:\s*(ADULT|MINOR|AMBIGUOUS)", text or "", re.IGNORECASE)
	return m.group(1).upper() if m else None


class OpenRouterReviewer(Reviewer):
	def __init__(self, api_key: str,
			first_pass_model: str = DEFAULT_FIRST_PASS_MODEL,
			escalation_model: str = DEFAULT_ESCALATION_MODEL,
			client: Optional[AsyncOpenAI] = None,
			timeout: float = 90.0,
			db_path=None):
		self.first_pass_model = first_pass_model
		self.escalation_model = escalation_model
		self._db_path = db_path
		self._client = client or AsyncOpenAI(
			api_key=api_key,
			base_url="https://openrouter.ai/api/v1",
			timeout=timeout,
			max_retries=2,
		)

	@classmethod
	def from_config(cls, config: dict, db_path=None) -> Optional["OpenRouterReviewer"]:
		key = config.get("openRouterToken") or config.get("openrouter_api_key")
		if not key:
			LOGGER.error("vision_review is enabled but no openRouterToken is set in config — reviewer will not run")
			return None
		return cls(
			api_key=str(key),
			first_pass_model=str(config.get("reviewer_model", DEFAULT_FIRST_PASS_MODEL)),
			escalation_model=str(config.get("reviewer_escalation_model", DEFAULT_ESCALATION_MODEL)),
			db_path=db_path,
		)

	def _recent_history(self, character: str) -> str:
		"""Compact summary of this character's recent review outcomes, used as
		evolving hints so the reviewer doesn't re-assess from scratch."""
		if not self._db_path or not character:
			return ""
		try:
			conn = sqlite3.connect(str(self._db_path))
			try:
				rows = conn.execute(
					"SELECT substr(digest, 1, 8), approved, category, attempt, substr(notes, 1, 140) "
					"FROM reviews WHERE character = ? ORDER BY id DESC LIMIT 8",
					(character,),
				).fetchall()
			finally:
				conn.close()
		except Exception as e:
			LOGGER.warning("Review-history lookup failed (%s); continuing without hints", e)
			return ""
		if not rows:
			return ""
		lines = []
		for digest8, approved, category, attempt, notes in reversed(rows):
			outcome = "approved" if approved else f"rejected ({category or 'quality'})"
			lines.append(f"- [{digest8} attempt {attempt}] {outcome}: {notes or '(no notes)'}")
		return (
			" REVIEW HISTORY for this character's recent generations (oldest first) — "
			"use it to spot recurring defects and to check whether previously-failed "
			"approaches were fixed: " + " | ".join(lines)
		)

	async def _chat(self, model: str, prompt: str, image_b64: str) -> str:
		resp = await self._client.chat.completions.create(
			model=model,
			max_tokens=500,
			messages=[{
				"role": "user",
				"content": [
					{"type": "text", "text": prompt},
					{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
				],
			}],
		)
		return (resp.choices[0].message.content or "").strip()

	async def review(self, image: bytes, prompt: str, attempt: int = 0,
			previous_notes: str = "", character: str = "") -> Verdict:
		image_b64 = base64.b64encode(image).decode()

		# Session context: on regenerations, tell the reviewer which attempt
		# this is and why the previous one was rejected, so it can verify the
		# fix instead of re-assessing blind. History hints add what has and
		# hasn't worked for this character recently.
		qc_prompt = QC_PROMPT
		history = await asyncio.to_thread(self._recent_history, character) if self._db_path else ""
		if history:
			qc_prompt += history
		if attempt > 0 and previous_notes:
			qc_prompt += (
				f" SESSION CONTEXT: this is regeneration attempt {attempt + 1} for the same "
				f"prompt. The previous attempt was rejected with this review: \"{previous_notes[:600]}\". "
				"Assess the current image normally, and specifically state whether the previous "
				"issues are resolved."
			)

		try:
			first = await self._chat(self.first_pass_model, qc_prompt, image_b64)
		except Exception as e:
			LOGGER.error("First-pass review failed (%s); approving (fail-open)", e)
			return Verdict(approved=True, notes=f"first-pass error: {e}")

		verdict = parse_verdict(first)
		if verdict == "GOOD":
			return Verdict(approved=True, notes=first[-400:])
		if verdict == "SKIP_MINOR":
			try:
				escalation = await self._chat(self.escalation_model, AGE_PROMPT, image_b64)
			except Exception as e:
				LOGGER.error("Age escalation failed (%s); refusing (fail-closed)", e)
				return Verdict(approved=False, category="minor",
					notes=f"first-pass flagged minor; escalation failed: {e}")
			assessment = parse_assessment(escalation)
			if assessment == "ADULT":
				LOGGER.info("Age check: first pass flagged, escalation cleared — approved")
				return Verdict(approved=True, notes=f"age cleared by {self.escalation_model}")
			# MINOR, AMBIGUOUS, or unparsable -> fail-closed
			return Verdict(approved=False, category="minor", notes=escalation[-400:])
		if verdict == "BAD":
			fix_add, fix_negate = parse_fix(first)
			return Verdict(approved=False, category="quality", notes=first[-400:],
				fix_add=fix_add, fix_negate=fix_negate)

		# No parsable verdict line: treat as a quality pass with a note
		LOGGER.warning("First-pass response had no VERDICT line; approving")
		return Verdict(approved=True, notes=first[-400:])
