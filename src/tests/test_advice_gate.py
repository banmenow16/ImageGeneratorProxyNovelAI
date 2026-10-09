"""Unit tests for the advice compliance gate: fact parsing and contract
judgment. The gate's law: the vision model reports neutral facts with an
attention cue, and CODE decides compliance — never the model."""

from imagine_generator_proxy.review.character import _parse_compliance_facts
from imagine_generator_proxy.server.advice import STAGES, _gate_violations

STAGE_BY_ID = {s["id"]: s for s in STAGES}


# --- fact parsing -----------------------------------------------------------

def test_parse_full_answer():
	facts = _parse_compliance_facts(
		"NIPPLES: drawn: areola and raised tip on both\n"
		"GARMENTS: none\n"
		"GENITALS: drawn: vulva with labia detail\n"
		"FACING: front\n"
		"BUTTOCKS: out of view")
	assert facts["nipples"] == "drawn"
	assert facts["genitals"] == "drawn"
	assert facts["facing"] == "front"
	assert facts["buttocks"] == "out_of_view"
	assert facts["garments"] == "none"


def test_parse_tail_covered_genitals():
	facts = _parse_compliance_facts(
		"NIPPLES: drawn\nGARMENTS: none\n"
		"GENITALS: covered: by the subject's tail\n"
		"FACING: front\nBUTTOCKS: out of view")
	assert facts["genitals"] == "covered"


def test_parse_featureless_and_partial_lines():
	facts = _parse_compliance_facts(
		"NIPPLES: absent: smooth featureless skin\n"
		"GARMENTS: none\n"
		"GENITALS: absent: featureless\n"
		"FACING: back\n"
		"BUTTOCKS: cleft drawn")
	assert facts["nipples"] == "absent"
	assert facts["genitals"] == "absent"
	assert facts["facing"] == "back"
	assert facts["buttocks"] == "cleft"


def test_parse_missing_lines_degrade_to_unreported():
	"""A truncated answer must not crash the gate — unreported facts only
	fail stages whose contract names them."""
	facts = _parse_compliance_facts("NIPPLES: drawn: both visible")
	assert facts["nipples"] == "drawn"
	assert facts["genitals"] == "unreported"
	assert facts["facing"] == "unreported"
	assert facts["buttocks"] == "unreported"
	assert facts["garments"] == ""


# --- contract judgment ------------------------------------------------------

def _facts(**over):
	base = {"nipples": "drawn", "nipples_raw": "drawn", "genitals": "drawn",
		"facing": "front", "buttocks": "out_of_view", "garments": "none"}
	base.update(over)
	return base


def test_nude_stage_passes_when_everything_visible():
	assert _gate_violations(_facts(), STAGE_BY_ID["nude"]) == ""


def test_nude_fails_on_featureless_genitals():
	violation = _gate_violations(_facts(genitals="absent"), STAGE_BY_ID["nude"])
	assert "genitals absent" in violation


def test_nude_fails_on_tail_covered_genitals():
	violation = _gate_violations(_facts(genitals="covered"), STAGE_BY_ID["nude"])
	assert "genitals covered" in violation


def test_nude_behind_requires_cleft_and_rear():
	stage = STAGE_BY_ID["nude_behind"]
	# three-quarter rear with cleft drawn passes
	assert _gate_violations(_facts(facing="side", buttocks="cleft"), stage) == ""
	# front-facing render fails even with cleft visible
	assert "facing front" in _gate_violations(_facts(facing="front",
		buttocks="cleft"), stage)
	# rear view but featureless fails
	assert "buttocks featureless" in _gate_violations(_facts(facing="back",
		buttocks="featureless"), stage)


def test_from_behind_fails_when_front_facing():
	stage = STAGE_BY_ID["from_behind"]
	assert "facing front" in _gate_violations(_facts(facing="front"), stage)
	assert _gate_violations(_facts(facing="back"), stage) == ""
	assert _gate_violations(_facts(facing="side"), stage) == ""


def test_topless_still_checks_nipples():
	stage = STAGE_BY_ID["topless"]
	assert "nipples absent" in _gate_violations(_facts(nipples="absent"), stage)


def test_portrait_never_gates():
	# portrait has no contract flags — facts present or not, no violation
	assert _gate_violations(_facts(facing="front", nipples="unreported"),
		STAGE_BY_ID["portrait"]) == ""


def test_gate_fail_open_on_no_facts():
	for stage in STAGES:
		assert _gate_violations(None, stage) == ""


# --- stage contract sanity ---------------------------------------------------

def test_stages_carry_nudity_and_rear_cues():
	# the compositions must name what the contract enforces, or the gate
	# rejects nearly every render (the featureless-breasts lesson)
	assert "nipples" in STAGE_BY_ID["topless"]["composition"]
	assert "pussy" in STAGE_BY_ID["nude"]["composition"]
	assert STAGE_BY_ID["nude"]["genitals"] is True
	assert "ass" in STAGE_BY_ID["nude_behind"]["composition"]
	assert STAGE_BY_ID["nude_behind"]["buttocks"] is True
	assert STAGE_BY_ID["from_behind"]["rear"] is True
	assert STAGE_BY_ID["nude_behind"]["rear"] is True


def test_keep_stage_outfit_is_not_a_violation():
	# from_behind keeps the canonical outfit — the gate must not flag it
	# (only rear orientation is enforced there)
	stage = STAGE_BY_ID["from_behind"]
	facts = _facts(facing="back", garments="black strapless bra, ruffled skirt")
	assert _gate_violations(facts, stage) == ""


def test_undressed_stage_flags_garments():
	violation = _gate_violations(_facts(garments="black tube top"),
		STAGE_BY_ID["topless"])
	assert "garments" in violation
