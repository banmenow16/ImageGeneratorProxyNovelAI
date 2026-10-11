"""Unit tests for the prompt massager (pure logic, no I/O except overlay tmp files)."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagine_generator_proxy.prompts import rules as prompt_rules
from imagine_generator_proxy.prompts import (
	MassagedPrompt,
	PromptMassager,
	build_negative_prompt,
	legacy_assemble,
)


@pytest.fixture
def massager() -> PromptMassager:
	return PromptMassager(list(prompt_rules.RULES), prompt_rules.RULES_VERSION)


def tags_of(effective: str) -> list:
	return [t.strip() for t in effective.split(",") if t.strip()]


# --- safety stage ---

@pytest.mark.parametrize("raw", [
	"cute loli, red dress",
	"shotacon art style",
	"a child playing in the park",
	"toddler on a swing",
	"girl age: 15, smiling",
	"aged 17, portrait",
	"17 years old",
	"1yo catgirl",
	"5-year-old dragon",
])
def test_unsafe_prompts_are_blocked(massager, raw):
	result = massager.massage(raw, "X")
	assert result.blocked_reason is not None
	assert result.blocked_rule is not None


@pytest.mark.parametrize("raw", [
	"woman age 25, smiling",
	"mature woman, 30 years old",
	"18 yo actress",
	"20 years old knight",
	"one girl, red dress, bedroom",
	"lollipop, candy shop background",
])
def test_safe_prompts_are_not_blocked(massager, raw):
	result = massager.massage(raw, "X")
	assert result.blocked_reason is None


def test_uniform_plus_childish_body_blocks(massager):
	result = massager.massage("school uniform, petite, classroom", "X")
	assert result.blocked_rule == "block-uniform-childish-body"


def test_uniform_flat_chest_blocks(massager):
	result = massager.massage("flat chest, school uniform", "X")
	assert result.blocked_rule == "block-uniform-childish-body"


def test_uniform_alone_is_stripped_not_blocked(massager):
	result = massager.massage("school uniform, red dress", "X")
	assert result.blocked_reason is None
	assert "school uniform" not in tags_of(result.effective)
	assert result.version == prompt_rules.RULES_VERSION


# --- cleanup stage ---

def test_dedupe_collapses_repeated_quality_tags(massager):
	raw = "red dress, score_9, source_anime, score_9, masterpiece, masterpiece"
	result = massager.massage(raw, "X")
	tags = tags_of(result.effective)
	assert tags.count("score_9") == 1
	assert tags.count("masterpiece") == 1
	assert "red dress" in tags


def test_weight_syntax_normalized_for_dedupe(massager):
	raw = "1.2::score_9::, red dress, score_9"
	result = massager.massage(raw, "X")
	# the weighted and plain spellings collapse to a single surviving tag
	survivors = [t for t in tags_of(result.effective) if "score_9" in t and "up" not in t]
	assert len(survivors) == 1


# --- negative-prompt leakage cleanup ---

def test_negative_marker_section_is_cut(massager):
	raw = "girl, red dress, intimate bedroom, negative: deformed, ugly, blurry, lowres, extra limbs"
	result = massager.massage(raw, "X")
	tags = tags_of(result.effective)
	assert "girl" in tags and "red dress" in tags
	for leaked in ("deformed", "ugly", "blurry", "lowres", "extra limbs"):
		assert leaked not in tags


def test_negative_prompt_marker_variant_is_cut(massager):
	raw = "girl, red dress, negative prompt: deformed, ugly, blurry"
	result = massager.massage(raw, "X")
	tags = tags_of(result.effective)
	assert "deformed" not in tags and "ugly" not in tags
	assert "red dress" in tags


def test_scene_suppressions_trailing_the_leak_are_cut_too(massager):
	# "outdoor, sky" after the marker were intended negatives; they must go
	raw = "girl, negative: deformed, ugly, blurry, lowres, outdoor, sky"
	result = massager.massage(raw, "X")
	tags = tags_of(result.effective)
	assert "outdoor" not in tags and "sky" not in tags


def test_bare_negative_vocab_run_is_cut(massager):
	raw = "girl, red dress, masterpiece, best quality, deformed, ugly, blurry, lowres, extra limbs, outdoor, sky"
	result = massager.massage(raw, "X")
	tags = tags_of(result.effective)
	assert "red dress" in tags
	for leaked in ("deformed", "ugly", "blurry", "lowres", "outdoor", "sky"):
		assert leaked not in tags


def test_short_negative_run_is_preserved(massager):
	# only a 2-run: conservative threshold must not fire
	raw = "girl, ugly, blurry, red dress"
	result = massager.massage(raw, "X")
	tags = tags_of(result.effective)
	assert "ugly" in tags and "blurry" in tags


def test_legit_outdoor_sky_without_run_is_preserved(massager):
	raw = "girl, outdoor, sky, sunset, picnic"
	result = massager.massage(raw, "X")
	tags = tags_of(result.effective)
	assert "outdoor" in tags and "sky" in tags and "sunset" in tags


# --- enhance stage ---

def test_quality_prefix_and_suffix_added_when_missing(massager):
	result = massager.massage("red dress", "X")
	tags = tags_of(result.effective)
	for tag in tags_of(prompt_rules.LEGACY_QUALITY_PREFIX):
		assert tag in tags
	for tag in tags_of(prompt_rules.V45_QUALITY_SUFFIX):
		assert tag in tags
	assert "red dress" in tags


def test_enhance_is_idempotent(massager):
	once = massager.massage("red dress, score_9", "X")
	twice = massager.massage(once.effective, "X")
	assert tags_of(once.effective) == tags_of(twice.effective)


# --- disabled / bypass paths ---

def test_disabled_massager_returns_legacy_assembly():
	m = PromptMassager(list(prompt_rules.RULES), prompt_rules.RULES_VERSION, enabled=False)
	result = m.massage("red dress", "X")
	assert result.effective == legacy_assemble("red dress")
	assert result.effective == prompt_rules.LEGACY_QUALITY_PREFIX + ",red dress"
	assert result.version is None


def test_safety_still_checkable_when_disabled():
	m = PromptMassager(list(prompt_rules.RULES), prompt_rules.RULES_VERSION, enabled=False)
	# massage() takes the legacy path...
	result = m.massage("cute loli", "X")
	assert result.blocked_reason is None
	# ...but check_safety() (used by the bypass path) still trips.
	assert m.check_safety("cute loli") is not None
	assert m.check_safety("red dress") is None


# --- negative prompt builder ---

def test_legacy_negative_prompt_is_byte_identical_with_historical_join():
	old_join = ', '.join([
		'text, watermark, signature, letters, writing, logo, advertisement, caption, error, emoji, chat, discord, message',
		'deformed, ugly, blurry, lowres, extra limbs, mutated hands, poor anatomy',
		'bad lighting, overexposed, underexposed, worst quality, low quality, jpeg artifacts, ugly, duplicate',
		'unnatural proportions',
		'',
	])
	# the bypass path (legacy=True) must reproduce the historical join exactly
	assert build_negative_prompt('', {}, legacy=True) == old_join


def test_massaged_negative_blocks_are_extended():
	# vision-QC-driven hardening of the hand/limb vocabulary
	massaged = build_negative_prompt('', {})
	for term in ('extra arms', 'extra digits', 'fused fingers', 'missing arms'):
		assert term in massaged


def test_negative_prompt_config_overrides_and_query_appends():
	assert build_negative_prompt('extra', {'negative_prompt': 'custom'}) == 'custom, extra'
	assert build_negative_prompt('extra', {'negative_prompt': ['a', 'b']}) == 'a, b, extra'


def test_negative_builder_extra_negations_between_blocks_and_query():
	out = build_negative_prompt('qn', {}, extra=['multiple girls', 'multiple boys'])
	assert 'multiple girls, multiple boys, qn' in out


# --- v3 enhance rules (vision-QC driven) ---

def test_large_chest_tags_kept_and_stabilized(massager):
	raw = "huge breasts, red dress, bedroom"
	result = massager.massage(raw, "X")
	tags = tags_of(result.effective)
	assert "huge breasts" in tags  # size tag untouched (user preference)
	assert "detailed anatomy" in tags
	assert "anatomically correct proportions" in tags


def test_no_chest_tags_no_stabilizers(massager):
	result = massager.massage("red dress, bedroom", "X")
	tags = tags_of(result.effective)
	assert "anatomically correct proportions" not in tags


def test_solo_guard_adds_person_negations(massager):
	result = massager.massage("solo, one girl, red dress", "X")
	assert "multiple girls" in result.negative_additions
	assert "multiple boys" in result.negative_additions


def test_no_solo_no_person_negations(massager):
	result = massager.massage("two girls on bed", "X")
	assert result.negative_additions == []


def test_contradictory_solo_prompt_does_not_fire_solo_guard(massager):
	"""SillyTavern cards send 'solo, one girl' even for two-figure scenes
	(experiment 22: 'carrying a small injured woman on her back'); the
	solo-guard must not suppress the intended second figure."""
	result = massager.massage("solo, one girl, carrying a small injured woman on her back", "X")
	assert result.negative_additions == []


def test_plain_solo_prompt_still_fires_solo_guard(massager):
	result = massager.massage("solo, one girl, red dress", "X")
	assert "multiple girls" in result.negative_additions


# --- reviewer fix sanitization (composition protection) ---

def test_sanitize_fix_tags_protects_multi_figure_scenes():
	from imagine_generator_proxy.prompts.massager import sanitize_fix_tags
	add, negate = sanitize_fix_tags(
		["solo", "adult woman", "arms at sides"],
		["extra characters", "background figures", "chibi", "multiple people"],
		"8ft tall woman carrying a small injured woman on her back")
	# composition-changing entries dropped; defect corrections kept
	assert add == ["adult woman", "arms at sides"]
	assert negate == ["chibi"]


def test_sanitize_fix_tags_keeps_suppressives_for_solo_prompts():
	from imagine_generator_proxy.prompts.massager import sanitize_fix_tags
	add, negate = sanitize_fix_tags(
		["arms at sides"],
		["background figures", "multiple people"],
		"solo, one girl, red dress, bedroom")
	assert add == ["arms at sides"]
	# genuine solo scene: suppressive negatives are appropriate
	assert negate == ["background figures", "multiple people"]


# --- overlay loading ---

OVERLAY_YAML = """\
version: 99
rules:
  - id: block-forbidden-word
    stage: safety
    severity: unsafe
    match:
      type: substring
      value: forbidden-word
    action:
      type: block
    notes: overlay rule
"""


def test_overlay_replaces_builtin_rules(tmp_path: Path):
	(tmp_path / "rules.yaml").write_text(OVERLAY_YAML, encoding="utf-8")
	m = PromptMassager.from_config({"rules_file": "rules.yaml"}, tmp_path)
	assert m.version == 99
	result = m.massage("contains forbidden-word here", "X")
	assert result.blocked_rule == "block-forbidden-word"
	# built-in safety rules are gone in overlay mode
	assert m.check_safety("cute loli") is None


def test_malformed_overlay_falls_back_to_builtins(tmp_path: Path):
	(tmp_path / "rules.yaml").write_text("version: [unclosed", encoding="utf-8")
	m = PromptMassager.from_config({"rules_file": "rules.yaml"}, tmp_path)
	assert m.version == prompt_rules.RULES_VERSION
	assert m.check_safety("cute loli") is not None


def test_missing_overlay_falls_back_to_builtins(tmp_path: Path):
	m = PromptMassager.from_config({"rules_file": "nope.yaml"}, tmp_path)
	assert m.version == prompt_rules.RULES_VERSION


def test_from_config_defaults():
	m = PromptMassager.from_config({}, Path("."))
	assert m.enabled is True
	assert m.safety_enabled is True
	assert m.version == prompt_rules.RULES_VERSION


# --- bench-measured rescue rules (guide benches 2/3) -----------------------

def test_nipples_cue_added_on_bare_chest(massager):
	result = massager.massage("1girl, solo, topless, bare breasts, sitting", "X")
	assert "nipples" in tags_of(result.effective)


def test_nipples_cue_not_added_when_variant_present(massager):
	result = massager.massage(
		"1girl, solo, topless, bare breasts, puffy nipples, sitting", "X")
	tags = tags_of(result.effective)
	assert tags.count("nipples") == 0  # puffy variant satisfies the contract


def test_nipples_cue_not_added_on_clothed_prompt(massager):
	result = massager.massage("1girl, solo, white shirt, sitting", "X")
	assert "nipples" not in tags_of(result.effective)


def test_genital_detail_phrase_added_with_pussy(massager):
	result = massager.massage(
		"1girl, solo, completely nude, pussy, standing", "X")
	assert "detailed realistic folds" in tags_of(result.effective)


def test_genital_detail_phrase_not_duplicated(massager):
	result = massager.massage(
		"1girl, solo, completely nude, pussy, detailed realistic folds", "X")
	assert tags_of(result.effective).count("detailed realistic folds") == 1


def test_gigantic_upgraded_to_enormous(massager):
	result = massager.massage("1girl, solo, gigantic breasts, standing", "X")
	tags = tags_of(result.effective)
	assert "gigantic breasts" not in tags
	assert "enormous breasts" in tags
