"""Unit tests for the prompt massager (pure logic, no I/O except overlay tmp files)."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagine_generator_proxy import prompt_rules
from imagine_generator_proxy.prompt_massager import (
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

def test_negative_prompt_is_byte_identical_with_legacy_join():
	old_join = ', '.join([
		'text, watermark, signature, letters, writing, logo, advertisement, caption, error, emoji, chat, discord, message',
		'deformed, ugly, blurry, lowres, extra limbs, mutated hands, poor anatomy',
		'bad lighting, overexposed, underexposed, worst quality, low quality, jpeg artifacts, ugly, duplicate',
		'unnatural proportions',
		'',
	])
	assert build_negative_prompt('', {}) == old_join


def test_negative_prompt_config_overrides_and_query_appends():
	assert build_negative_prompt('extra', {'negative_prompt': 'custom'}) == 'custom, extra'
	assert build_negative_prompt('extra', {'negative_prompt': ['a', 'b']}) == 'a, b, extra'


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
