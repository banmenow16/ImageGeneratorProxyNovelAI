"""Rule-based prompt massager: safety filter -> cleanup -> enhance.

Pure logic, no I/O (the only exception is from_config, which reads the
optional rules overlay). The handler calls massage() when massage mode is on
and check_safety() when it is bypassed, so the safety stage can never be
skipped by a toggle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from imagine_generator_proxy.prompts import rules as prompt_rules

LOGGER = logging.getLogger(__name__)

# NovelAI weight syntax: "1.2::tag::"
_WEIGHT_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*::\s*(.+?)\s*::\s*$")

# Substrings indicating a multi-figure scene even when 'solo'/'one girl'
# tags are present (SillyTavern character cards send both — experiment 22:
# "solo, one girl" alongside "carrying a small injured woman on her back").
_MULTI_FIGURE_HINTS = (
	"two ", "carrying", "another ", "second ", "both women", "both girls",
	"both men", "pair ", "hugging", "embracing", "kissing", "holding her",
	"holding him", "piggyback", "princess carry", "helping ",
)

# Reviewer fix tags that would change WHO is in the image. Fixes must
# correct defects in existing figures (hands, torso, age), never rewrite
# composition — experiment 22's fix ("solo, neg:extra characters") deleted
# the second woman the scene existed to show.
_COMPOSITION_POSITIVE = {
	"solo", "one girl", "1girl", "one boy", "1boy", "alone", "solo focus",
	"multiple girls", "multiple boys", "two girls", "two women", "two men",
}
_SUPPRESSIVE_NEGATIVE = {
	"multiple girls", "multiple boys", "multiple people", "extra characters",
	"extra figures", "extra people", "background figures", "background people",
	"second figure", "second person", "second woman", "second man",
	"second girl", "second boy",
}


def looks_multi_figure(prompt: str) -> bool:
	"""True when scene language implies more than one figure."""
	lower = (prompt or "").lower()
	return any(hint in lower for hint in _MULTI_FIGURE_HINTS)


def sanitize_fix_tags(fix_add: list, fix_negate: list, prompt: str) -> tuple:
	"""Filter reviewer-proposed fixes so they cannot change composition.

	Positive figure-count tags are always dropped. Suppressive negatives are
	dropped for multi-figure scenes (they delete intended characters) but
	kept for genuine solo prompts (they remove photobombers).
	"""
	add = [t for t in fix_add if t.lower() not in _COMPOSITION_POSITIVE]
	if looks_multi_figure(prompt):
		negate = [t for t in fix_negate if t.lower() not in _SUPPRESSIVE_NEGATIVE]
	else:
		negate = list(fix_negate)
	return add, negate


@dataclass
class MassagedPrompt:
	"""Result of massaging a raw prompt.

	version None means the legacy/bypass path: `effective` is the historical
	assembly (LEGACY_QUALITY_PREFIX + "," + raw) and the digest must use the
	legacy formula.
	"""

	original: str
	effective: str
	version: Optional[int]
	blocked_reason: Optional[str] = None
	blocked_rule: Optional[str] = None
	applied_rules: List[str] = field(default_factory=list)
	removed: List[str] = field(default_factory=list)
	negative_additions: List[str] = field(default_factory=list)


# The historical blocks, frozen verbatim: the massage-bypass path must keep
# producing byte-identical negatives (and therefore legacy digests) so it
# still hits the pre-massager cache. Never edit this list.
LEGACY_NEGATIVE_BLOCKS = [
	"text, watermark, signature, letters, writing, logo, advertisement, caption, error, emoji, chat, discord, message",
	"deformed, ugly, blurry, lowres, extra limbs, mutated hands, poor anatomy",
	"bad lighting, overexposed, underexposed, worst quality, low quality, jpeg artifacts, ugly, duplicate",
	"unnatural proportions",
]

# Blocks for the massaged path. Vision QC of the cache showed the legacy
# hand/limb vocabulary was too weak (fused fingers and extra arms still
# render). NOTE: any change here changes the effective negative prompt and
# therefore the massaged digest — bump prompt_rules.RULES_VERSION alongside.
DEFAULT_NEGATIVE_BLOCKS = [
	"text, watermark, signature, letters, writing, logo, advertisement, caption, error, emoji, chat, discord, message",
	"deformed, ugly, blurry, lowres, extra limbs, extra arms, extra digits, fused fingers, missing arms, mutated hands, poor anatomy",
	"bad lighting, overexposed, underexposed, worst quality, low quality, jpeg artifacts, ugly, duplicate",
	"unnatural proportions",
]


def build_negative_prompt(query_negative: str, config: dict, extra: Optional[List[str]] = None,
		legacy: bool = False) -> str:
	"""Build the negative prompt.

	legacy=True freezes the historical blocks so the bypass path stays
	byte-identical with pre-massager generations (trailing ", " included).
	Config `negative_prompt` (str or list of str) overrides the built-in
	blocks on the massaged path only; rule-generated `extra` negations are
	appended after the blocks, before the query negative.
	"""
	blocks = config.get("negative_prompt")
	if blocks is None:
		blocks = LEGACY_NEGATIVE_BLOCKS if legacy else DEFAULT_NEGATIVE_BLOCKS
	if isinstance(blocks, str):
		blocks = [blocks]
	return ", ".join(list(blocks) + list(extra or []) + [query_negative])


def legacy_assemble(raw_prompt: str) -> str:
	"""Historical prompt assembly, kept byte-identical for the bypass path."""
	return prompt_rules.LEGACY_QUALITY_PREFIX + "," + raw_prompt


def _split_tags(prompt: str) -> List[str]:
	return [t for t in (s.strip() for s in prompt.split(",")) if t]


def _normalize_tag(tag: str) -> Tuple[str, float]:
	"""Return (normalized_tag, weight), unwrapping '0.8::tag::' weights."""
	m = _WEIGHT_RE.match(tag)
	if m:
		try:
			weight = float(m.group(1))
		except ValueError:
			weight = 1.0
		return (m.group(2).strip().lower(), weight)
	return (tag.lower(), 1.0)


def _norm(tag: str) -> str:
	return _normalize_tag(tag)[0]


_LEAK_RULE_IDS = ("strip-negative-section", "strip-negative-vocab-run")


def strip_negative_leak(prompt: str) -> str:
	"""Apply ONLY the negative-leak cleanup rules (literal 'negative:' section
	cut and bare negative-vocabulary run cut). Used where raw prompts are
	displayed or prefilled: the leak is a known caller artifact, and showing
	it (e.g. in the workbench positive field) reads as a bug. No enhance
	additions, no safety behavior — this is a display cleanup."""
	leak_rules = [r for r in prompt_rules.RULES
		if r.get("id") in _LEAK_RULE_IDS and r.get("stage") == "cleanup"]
	if not leak_rules:
		return prompt
	tags = [(t, _norm(t)) for t in _split_tags(prompt)]
	PromptMassager(leak_rules, version=None)._run_stage("cleanup", tags, [], [])
	return ", ".join(display for display, _ in tags)


class PromptMassager:
	def __init__(self, rules: List[dict], version: Optional[int], enabled: bool = True, safety_enabled: bool = True):
		self.rules = rules
		self.version = version
		self.enabled = enabled
		self.safety_enabled = safety_enabled

	@classmethod
	def from_config(cls, config: dict, config_dir: Path,
			backend_overrides: Optional[dict] = None) -> "PromptMassager":
		enabled = bool(config.get("prompt_massage", True))
		safety_enabled = bool(config.get("safety_filter", True))
		rules = list(prompt_rules.RULES)
		version = prompt_rules.RULES_VERSION

		# generator-provided overrides: per-model quality vocabulary and
		# quirks (see ImageGenerator.rules_overrides). Universal rules —
		# safety, leak-cut, dedupe, taste calibration — stay shared.
		if backend_overrides:
			for rule_id in backend_overrides.get("remove", []):
				rules = [r for r in rules if r.get("id") != rule_id]
			for rule_id, replacement in (backend_overrides.get("replace") or {}).items():
				rules = [replacement if r.get("id") == rule_id else r for r in rules]
			rules += list(backend_overrides.get("add", []))

		rules_file = config.get("rules_file")
		if rules_file:
			overlay = cls._load_overlay(Path(rules_file), config_dir)
			if overlay is not None:
				overlay_rules, overlay_version = overlay
				rules = overlay_rules
				version = overlay_version
				LOGGER.info(
					"Loaded rules overlay (%s rules, version %s); built-in rules are overridden",
					len(rules), version,
				)
		return cls(rules, version, enabled=enabled, safety_enabled=safety_enabled)

	@staticmethod
	def _load_overlay(rules_file: Path, config_dir: Path) -> Optional[Tuple[List[dict], int]]:
		path = rules_file if rules_file.is_absolute() else config_dir / rules_file
		try:
			import yaml
		except Exception:
			LOGGER.error("Config sets rules_file=%s but PyYAML is not installed; using built-in rules", rules_file)
			return None
		try:
			data = yaml.safe_load(path.read_text(encoding="utf-8"))
			if not isinstance(data, dict) or "rules" not in data or "version" not in data:
				raise ValueError("overlay must be a mapping with 'version' and 'rules'")
			LOGGER.warning("Rules overlay %s replaces all built-in rules (including safety) — it must define its own", path)
			return (list(data["rules"]), int(data["version"]))
		except Exception as e:
			LOGGER.error("Failed to load rules overlay %s: %s — using built-in rules", path, e)
			return None

	# --- matching ---

	def _matches(self, spec: dict, tags_norm: set, prompt_lower: str) -> bool:
		t = spec.get("type")
		if t == "always":
			return True
		if t == "tag":
			return _norm(str(spec["value"])) in tags_norm
		if t == "tags":
			return any(_norm(str(v)) in tags_norm for v in spec["value"])
		if t == "solo_scene":
			# solo tags present AND the scene language doesn't describe a
			# second figure (SillyTavern cards send contradictory prompts)
			any_solo = any(_norm(str(v)) in tags_norm for v in spec["value"])
			return any_solo and not looks_multi_figure(prompt_lower)
		if t == "substring":
			return str(spec["value"]).lower() in prompt_lower
		if t == "regex":
			flags = re.IGNORECASE if "i" in (spec.get("flags") or "") else 0
			return re.search(str(spec["value"]), prompt_lower, flags) is not None
		if t == "all":
			return all(self._matches(s, tags_norm, prompt_lower) for s in spec["values"])
		if t == "any":
			return any(self._matches(s, tags_norm, prompt_lower) for s in spec["values"])
		LOGGER.warning("Unknown match type %r in rule spec %r", t, spec)
		return False

	# --- actions ---

	def _apply(self, rule: dict, tags: List[Tuple[str, str]], applied: List[str],
			removed: List[str], negations: List[str]) -> None:
		action = rule.get("action") or {}
		spec = rule.get("match") or {}
		atype = action.get("type")
		value = action.get("value")

		if atype == "strip":
			targets = self._spec_tag_values(spec)
			before = len(tags)
			tags[:] = [(d, n) for (d, n) in tags if n not in targets]
			if len(tags) != before:
				removed.append(rule["id"])
		elif atype == "truncate_at_match":
			# Cut everything from the first tag whose text matches the rule's
			# regex (e.g. a literal "negative:" section head) to the end.
			flags = re.IGNORECASE if "i" in (spec.get("flags") or "") else 0
			pattern = re.compile(str(spec.get("value", "")), flags)
			for idx, (display, _) in enumerate(tags):
				if pattern.search(display):
					removed.extend(d for d, _ in tags[idx:])
					del tags[idx:]
					break
		elif atype == "strip_negvocab_run":
			# Cut from the earliest run of `run_length` consecutive
			# negative-vocabulary tags to the end (leaked negative section
			# without a marker). Runs shorter than run_length are preserved.
			vocab = {_norm(str(v)) for v in (value or [])}
			run_length = int(action.get("run_length", 3))
			for idx in range(0, len(tags) - run_length + 1):
				if all(n in vocab for _, n in tags[idx:idx + run_length]):
					removed.extend(d for d, _ in tags[idx:])
					del tags[idx:]
					break
		elif atype == "replace":
			targets = self._spec_tag_values(spec)
			replacement = [(t, _norm(t)) for t in _split_tags(str(value))]
			new_tags: List[Tuple[str, str]] = []
			for display, norm in tags:
				if norm in targets:
					new_tags.extend(replacement)
				else:
					new_tags.append((display, norm))
			tags[:] = new_tags
			applied.append(rule["id"])
		elif atype in ("prepend", "append", "ensure_prepend", "ensure_append"):
			additions = [(t, _norm(t)) for t in _split_tags(str(value))]
			if atype.startswith("ensure_"):
				present = {n for _, n in tags}
				additions = [(d, n) for (d, n) in additions if n not in present]
			if additions:
				if atype.endswith("prepend"):
					tags[:] = additions + tags
				else:
					tags.extend(additions)
				applied.append(rule["id"])
		elif atype == "dedupe":
			targets = self._spec_tag_values(spec)
			seen = set()
			new_tags = []
			for display, norm in tags:
				if norm in targets:
					if norm in seen:
						removed.append(display)
						continue
					seen.add(norm)
				new_tags.append((display, norm))
			tags[:] = new_tags
		elif atype == "negate":
			# Add prompt-specific negations (e.g. "multiple boys" for solo
			# prompts) to the negative prompt via MassagedPrompt.negative_additions.
			for tag in _split_tags(str(value)):
				if tag not in negations:
					negations.append(tag)
			applied.append(rule["id"])
		elif atype == "block":
			# Block severity is handled by the safety stage; nothing to do here.
			pass
		else:
			LOGGER.warning("Unknown action type %r in rule %r", atype, rule.get("id"))

	@staticmethod
	def _spec_tag_values(spec: dict) -> set:
		if spec.get("type") == "tag":
			return {_norm(str(spec["value"]))}
		if spec.get("type") == "tags":
			return {_norm(str(v)) for v in spec["value"]}
		return set()

	# --- pipeline ---

	def _run_stage(
		self, stage: str, tags: List[Tuple[str, str]], applied: List[str], removed: List[str],
		negations: Optional[List[str]] = None,
	) -> Optional[Tuple[str, str]]:
		"""Run the rules of one stage over the tag list (mutated in place).

		Returns (rule_id, reason) as soon as an unsafe rule matches.
		"""
		negations = negations if negations is not None else []
		for rule in self.rules:
			if rule.get("stage") != stage:
				continue
			spec = rule.get("match") or {}
			# Match against the current state of the prompt, not the raw input.
			current_lower = ", ".join(d for d, _ in tags).lower()
			tags_norm = {n for _, n in tags}
			if not self._matches(spec, tags_norm, current_lower):
				continue
			if rule.get("severity") == "unsafe":
				return (rule.get("id", "unknown"), rule.get("notes") or "blocked by safety rule")
			self._apply(rule, tags, applied, removed, negations)
		return None

	def check_safety(self, raw_prompt: str) -> Optional[Tuple[str, str]]:
		"""Return (rule_id, reason) if an unsafe rule trips on the raw prompt.

		Intended for the massage-bypass path so the safety filter still runs.
		"""
		if not self.safety_enabled:
			return None
		tags = [(t, _norm(t)) for t in _split_tags(raw_prompt)]
		return self._run_stage("safety", tags, [], [])

	def massage(self, raw_prompt: str, character: str) -> MassagedPrompt:
		"""Run the full pipeline. `character` is reserved for per-character
		rules (analysis-driven); unused in v1."""
		if not self.enabled:
			return MassagedPrompt(
				original=raw_prompt, effective=legacy_assemble(raw_prompt), version=None
			)

		tags = [(t, _norm(t)) for t in _split_tags(raw_prompt)]
		applied: List[str] = []
		removed: List[str] = []
		negations: List[str] = []

		if self.safety_enabled:
			blocked = self._run_stage("safety", tags, applied, removed)
			if blocked is not None:
				return MassagedPrompt(
					original=raw_prompt, effective=raw_prompt, version=self.version,
					blocked_reason=blocked[1], blocked_rule=blocked[0],
					applied_rules=applied, removed=removed,
				)

		self._run_stage("cleanup", tags, applied, removed, negations)
		self._run_stage("enhance", tags, applied, removed, negations)

		effective = ", ".join(display for display, _ in tags)
		return MassagedPrompt(
			original=raw_prompt, effective=effective, version=self.version,
			applied_rules=applied, removed=removed, negative_additions=negations,
		)
