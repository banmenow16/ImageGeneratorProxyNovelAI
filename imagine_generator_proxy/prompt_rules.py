"""Rule data for the prompt massager. This module contains data only — the
pipeline logic lives in prompt_massager.py.

Built-in rules are plain Python (not YAML) so that a malformed rule file can
never silently disable the safety stage: an error here crashes server startup
loudly. Users can override the whole list via an optional `rules_file` overlay
(see PromptMassager.from_config), which must carry its own top-level `version`.

Rule schema:
	id        stable identifier (logged, returned on block)
	stage     safety | cleanup | enhance — fixed pipeline order
	severity  safe | unsafe — unsafe + match => refuse the request
	match     {type: tag|tags|substring|regex|all|any|always, ...}
	action    {type: block|strip|replace|prepend|append|ensure_prepend|ensure_append|dedupe, value}
	notes     human context (used as the block reason)
"""

from __future__ import annotations

RULES_VERSION: int = 2

LEGACY_QUALITY_PREFIX = "score_9, score_8_up, score_7_up, score_6_up, source_anime"
V45_QUALITY_SUFFIX = "location, very aesthetic, masterpiece, no text"

# Quality-defect vocabulary that belongs in the NEGATIVE prompt. Found leaked
# into the tail of positive prompts in ~6% of the cached corpus (SillyTavern
# character cards appending their negative block). Used by the run-detector:
# a consecutive run of >= 3 of these tags is treated as a leaked negative
# section and cut, together with any scene suppressions trailing it.
NEGATIVE_VOCAB = [
	"deformed", "ugly", "blurry", "lowres", "extra limbs", "mutated hands",
	"poor anatomy", "bad lighting", "overexposed", "underexposed",
	"worst quality", "low quality", "jpeg artifacts", "text", "watermark",
	"signature", "logo", "duplicate", "duplicates", "extra fingers",
	"emoji", "letters", "writing", "advertisement", "caption", "error",
	"chat", "discord", "message", "unnatural proportions",
]

RULES = [
	# --- stage: safety (runs first, always) ---
	{"id": "block-loli-shota", "stage": "safety", "severity": "unsafe",
		"match": {"type": "regex", "value": r"\b(loli|shota)(con)?\b", "flags": "i"},
		"action": {"type": "block"}, "notes": "minor-coded terms are not permitted"},
	{"id": "block-child-words", "stage": "safety", "severity": "unsafe",
		"match": {"type": "regex", "value": r"\b(child|children|kid|toddler|infant|underage|preteen|pre-teen)\b", "flags": "i"},
		"action": {"type": "block"}, "notes": "child-related terms are not permitted"},
	{"id": "block-age-under-18", "stage": "safety", "severity": "unsafe",
		"match": {"type": "regex", "value": r"\b(?:aged?|age)\s*[:]?\s*([0-9]|1[0-7])\b|\b([0-9]|1[0-7])[-\s]*(?:yo|y/o|years?[- ]old)\b", "flags": "i"},
		"action": {"type": "block"}, "notes": "explicit age under 18"},
	# Combo rules MUST run before the corresponding strip rule below, otherwise
	# the strip removes the tag this rule needs to see.
	{"id": "block-uniform-childish-body", "stage": "safety", "severity": "unsafe",
		"match": {"type": "all", "values": [
			{"type": "tag", "value": "school uniform"},
			{"type": "any", "values": [
				{"type": "tag", "value": "petite"},
				{"type": "tag", "value": "flat chest"},
			]},
		]},
		"action": {"type": "block"}, "notes": "school uniform combined with a childish body type"},
	{"id": "strip-school-uniform", "stage": "safety", "severity": "safe",
		"match": {"type": "tag", "value": "school uniform"},
		"action": {"type": "strip"},
		"notes": "uniform alone is stripped after the combo check above"},

	# --- stage: cleanup ---
	# The caller sometimes appends its negative prompt to the positive one
	# (observed as "negative: ..." / "negative prompt: ..." sections). Everything
	# after the marker is intended-negative, so cut it. Scene-specific
	# suppressions after the marker (e.g. "outdoor, sky") are discarded with
	# it; extracting them into the real negative prompt is a future refinement.
	{"id": "strip-negative-section", "stage": "cleanup", "severity": "safe",
		"match": {"type": "regex", "value": r"negative\s+prompt\s*:|negative\s*:", "flags": "i"},
		"action": {"type": "truncate_at_match"},
		"notes": "cut literal 'negative:' section appended by the caller"},
	{"id": "strip-negative-vocab-run", "stage": "cleanup", "severity": "safe",
		"match": {"type": "always"},
		"action": {"type": "strip_negvocab_run", "value": NEGATIVE_VOCAB, "run_length": 3},
		"notes": "cut trailing runs of >= 3 negative-vocabulary tags with no marker"},

	{"id": "dedupe-quality-tags", "stage": "cleanup", "severity": "safe",
		"match": {"type": "tags", "value": [
			"score_9", "score_8_up", "score_7_up", "score_6_up", "score_5_up",
			"source_anime", "best quality", "very aesthetic", "absurdres",
			"masterpiece", "location", "no text", "amazing quality",
		]},
		"action": {"type": "dedupe"},
		"notes": "collapse quality tags duplicated by caller + prefix"},

	# --- stage: enhance ---
	{"id": "ensure-quality-prefix", "stage": "enhance", "severity": "safe",
		"match": {"type": "always"},
		"action": {"type": "ensure_prepend", "value": LEGACY_QUALITY_PREFIX},
		"notes": "canonical quality prefix, added only when missing"},
	{"id": "ensure-v45-suffix", "stage": "enhance", "severity": "safe",
		"match": {"type": "always"},
		"action": {"type": "ensure_append", "value": V45_QUALITY_SUFFIX},
		"notes": "replaces the library quality_toggle, which is disabled when massage is on"},
]
