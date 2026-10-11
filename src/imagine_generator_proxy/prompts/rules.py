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

import json
from pathlib import Path

RULES_VERSION: int = 5

# Bulky vocabularies live in rule_data.json (loaded here, loud failure if the
# file is missing or malformed — same fail-closed philosophy as the rules).
# Editing that file changes massaged prompts: bump RULES_VERSION alongside.
_DATA_PATH = Path(__file__).parent / "rule_data.json"
with _DATA_PATH.open("r", encoding="utf-8") as _f:
	DATA = json.load(_f)

# Large-chest size tags (kept verbatim — user preference). Vision QC found
# these correlate with melted torso silhouettes and lost/untraceable arms;
# the rule adds stability tags instead of touching the size tags.
LARGE_CHEST_TAGS = DATA["large_chest_tags"]

# Solo indicators: vision QC saw "solo" prompts gain background figures the
# standard negative blocks don't suppress (they only cover "multiple girls").
SOLO_TAGS = DATA["solo_tags"]

LEGACY_QUALITY_PREFIX = "score_9, score_8_up, score_7_up, score_6_up, source_anime"
V45_QUALITY_SUFFIX = "location, very aesthetic, masterpiece, no text"

# Quality-defect vocabulary that belongs in the NEGATIVE prompt. Found leaked
# into the tail of positive prompts in ~6% of the cached corpus (SillyTavern
# character cards appending their negative block). Used by the run-detector:
# a consecutive run of >= 3 of these tags is treated as a leaked negative
# section and cut, together with any scene suppressions trailing it.
NEGATIVE_VOCAB = DATA["negative_vocab"]

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
		"match": {"type": "tags", "value": DATA["dedupe_quality_tags"]},
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
	{"id": "stabilize-large-chest", "stage": "enhance", "severity": "safe",
		"match": {"type": "tags", "value": LARGE_CHEST_TAGS},
		"action": {"type": "ensure_append", "value": "detailed anatomy, anatomically correct proportions"},
		"notes": "large-chest prompts trend to melted torsos; stabilize without touching the size tags"},

	# --- vision-QC-informed tag strengthening (from Mirror findings) ---
	{"id": "upgrade-small-breasts", "stage": "enhance", "severity": "safe",
		"match": {"type": "tag", "value": "small breasts"},
		"action": {"type": "replace", "value": "flat chest"},
		"notes": "small breasts is too weak vs the model default; flat chest is stronger"},
	{"id": "blue-tint-counter", "stage": "enhance", "severity": "safe",
		"match": {"type": "tag", "value": "black hair"},
		"action": {"type": "append", "value": "matte hair"},
		"notes": "black hair gets blue sheen; matte hair counters the reflective tint (38 QC rounds)"},
	{"id": "negate-blue-hair-with-black", "stage": "enhance", "severity": "safe",
		"match": {"type": "tag", "value": "black hair"},
		"action": {"type": "negate", "value": "blue hair, teal hair"},
		"notes": "preventive: black hair often renders with blue/teal tint"},
	{"id": "upgrade-dark-pink", "stage": "enhance", "severity": "safe",
		"match": {"type": "tag", "value": "dark pink hair"},
		"action": {"type": "replace", "value": "rose pink hair"},
		"notes": "dark pink shifts to purple; rose pink stays closer to intent (11 QC rounds)"},
	# --- bench-measured existence-cue rescues (guide benches 2/3, 2026-10-09) ---
	{"id": "ensure-nipples-on-bare-chest", "stage": "enhance", "severity": "safe",
		"match": {"type": "all", "values": [
			{"type": "any", "values": [
				{"type": "tag", "value": "bare breasts"},
				{"type": "tag", "value": "topless"},
				{"type": "tag", "value": "nude"},
				{"type": "tag", "value": "completely nude"},
			]},
			{"type": "none", "values": [
				{"type": "tag", "value": "nipples"},
				{"type": "tag", "value": "puffy nipples"},
				{"type": "tag", "value": "puffy erect nipples"},
				{"type": "tag", "value": "erect nipples"},
				{"type": "tag", "value": "swollen nipples"},
				{"type": "tag", "value": "inverted nipples"},
				{"type": "tag", "value": "long nipples"},
				{"type": "tag", "value": "big nipples"},
				{"type": "tag", "value": "huge nipples"},
			]},
		]},
		"action": {"type": "ensure_append", "value": "nipples"},
		"notes": "v4.5 draws featureless chests by default — the cue controls WHETHER "
			"nipples render (8/8 with cue on clean identities; uncued pre-fix runs never rendered)"},
	{"id": "ensure-genital-detail-phrase", "stage": "enhance", "severity": "safe",
		"match": {"type": "all", "values": [
			{"type": "any", "values": [
				{"type": "tag", "value": "pussy"},
				{"type": "tag", "value": "spread pussy"},
			]},
			{"type": "none", "values": [
				{"type": "tag", "value": "detailed realistic folds"},
			]},
		]},
		"action": {"type": "ensure_append", "value": "detailed realistic folds"},
		"notes": "the detail phrase carries the genital render: bare `pussy` is a coin flip "
			"(1/2), `pussy, detailed realistic folds` rendered 2/2 with full detail"},
	{"id": "upgrade-gigantic-breasts", "stage": "enhance", "severity": "safe",
		"match": {"type": "tag", "value": "gigantic breasts"},
		"action": {"type": "replace", "value": "enormous breasts"},
		"notes": "gigantic is a rare word that rendered BELOW huge (bench 1) and tied "
			"enormous once by noise (bench 2); enormous is the measured ceiling"},
	{"id": "solo-guard", "stage": "enhance", "severity": "safe",
		"match": {"type": "solo_scene", "value": SOLO_TAGS},
		"action": {"type": "negate", "value": "multiple girls, multiple boys"},
		"notes": "solo prompts gain background figures; suppressed only when the scene is genuinely single-figure (SillyTavern cards send 'solo, one girl' even for two-figure scenes)"},
]
