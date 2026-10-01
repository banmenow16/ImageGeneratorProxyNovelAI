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

from imagine_generator_proxy import prompt_rules

LOGGER = logging.getLogger(__name__)

# NovelAI weight syntax: "1.2::tag::"
_WEIGHT_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*::\s*(.+?)\s*::\s*$")


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


DEFAULT_NEGATIVE_BLOCKS = [
	"text, watermark, signature, letters, writing, logo, advertisement, caption, error, emoji, chat, discord, message",
	"deformed, ugly, blurry, lowres, extra limbs, mutated hands, poor anatomy",
	"bad lighting, overexposed, underexposed, worst quality, low quality, jpeg artifacts, ugly, duplicate",
	"unnatural proportions",
]


def build_negative_prompt(query_negative: str, config: dict) -> str:
	"""Build the negative prompt.

	Config `negative_prompt` (str or list of str) replaces the built-in
	blocks; the query negative is always appended — including when empty, to
	stay byte-identical with the historical join (trailing ", ").
	"""
	blocks = config.get("negative_prompt") or DEFAULT_NEGATIVE_BLOCKS
	if isinstance(blocks, str):
		blocks = [blocks]
	return ", ".join(list(blocks) + [query_negative])


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


class PromptMassager:
	def __init__(self, rules: List[dict], version: Optional[int], enabled: bool = True, safety_enabled: bool = True):
		self.rules = rules
		self.version = version
		self.enabled = enabled
		self.safety_enabled = safety_enabled

	@classmethod
	def from_config(cls, config: dict, config_dir: Path) -> "PromptMassager":
		enabled = bool(config.get("prompt_massage", True))
		safety_enabled = bool(config.get("safety_filter", True))
		rules = list(prompt_rules.RULES)
		version = prompt_rules.RULES_VERSION

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

	def _apply(self, rule: dict, tags: List[Tuple[str, str]], applied: List[str], removed: List[str]) -> None:
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
		self, stage: str, tags: List[Tuple[str, str]], applied: List[str], removed: List[str]
	) -> Optional[Tuple[str, str]]:
		"""Run the rules of one stage over the tag list (mutated in place).

		Returns (rule_id, reason) as soon as an unsafe rule matches.
		"""
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
			self._apply(rule, tags, applied, removed)
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

		if self.safety_enabled:
			blocked = self._run_stage("safety", tags, applied, removed)
			if blocked is not None:
				return MassagedPrompt(
					original=raw_prompt, effective=raw_prompt, version=self.version,
					blocked_reason=blocked[1], blocked_rule=blocked[0],
					applied_rules=applied, removed=removed,
				)

		self._run_stage("cleanup", tags, applied, removed)
		self._run_stage("enhance", tags, applied, removed)

		effective = ", ".join(display for display, _ in tags)
		return MassagedPrompt(
			original=raw_prompt, effective=effective, version=self.version,
			applied_rules=applied, removed=removed,
		)
