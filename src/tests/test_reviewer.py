"""Unit tests for the two-tier OpenRouter reviewer (stubbed client)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from imagine_generator_proxy.review.openrouter import (
	OpenRouterReviewer,
	parse_assessment,
	parse_verdict,
)
from imagine_generator_proxy.review.base import Verdict


# --- parsing ---

@pytest.mark.parametrize("text,expected", [
	("1. stuff\n2. stuff\nVERDICT: GOOD", "GOOD"),
	("VERDICT: BAD: fused fingers", "BAD"),
	("report...\nVERDICT: SKIP_MINOR", "SKIP_MINOR"),
	("verdict: good", "GOOD"),  # case-insensitive
	("no verdict here", None),
	("", None),
])
def test_parse_verdict(text, expected):
	assert parse_verdict(text) == expected


@pytest.mark.parametrize("text,expected", [
	("ASSESSMENT: ADULT", "ADULT"),
	("analysis...\nASSESSMENT: AMBIGUOUS", "AMBIGUOUS"),
	("ASSESSMENT: MINOR", "MINOR"),
	("nothing", None),
])
def test_parse_assessment(text, expected):
	assert parse_assessment(text) == expected


# --- stubbed two-tier logic ---

class _StubCompletions:
	def __init__(self, responses):
		self._responses = list(responses)
		self.calls = []
		self.prompts = []

	async def create(self, **kwargs):
		self.calls.append(kwargs.get("model"))
		self.prompts.append(kwargs["messages"][0]["content"][0]["text"])
		if not self._responses:
			raise AssertionError("stub exhausted")
		resp = self._responses.pop(0)
		if isinstance(resp, Exception):
			raise resp
		return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=resp))])


def make_reviewer(responses):
	completions = _StubCompletions(responses)
	client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
	reviewer = OpenRouterReviewer(api_key="test-key", client=client)
	return reviewer, completions


def run(coro):
	return asyncio.run(coro)


def test_good_first_pass_approves_without_escalation():
	reviewer, completions = make_reviewer(["QC report\nVERDICT: GOOD"])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is True
	assert verdict.category is None
	assert len(completions.calls) == 1


def test_bad_first_pass_rejects_on_quality():
	reviewer, _ = make_reviewer(["QC report\nVERDICT: BAD: extra arms"])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is False
	assert verdict.category == "quality"


def test_minor_flag_cleared_by_escalation():
	reviewer, completions = make_reviewer([
		"VERDICT: SKIP_MINOR",
		"looks stylized but adult\nASSESSMENT: ADULT",
	])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is True
	assert len(completions.calls) == 2  # first pass + escalation


def test_minor_flag_confirmed_by_escalation():
	reviewer, _ = make_reviewer([
		"VERDICT: SKIP_MINOR",
		"ASSESSMENT: MINOR",
	])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is False
	assert verdict.category == "minor"


def test_ambiguous_age_fails_closed():
	reviewer, _ = make_reviewer([
		"VERDICT: SKIP_MINOR",
		"ASSESSMENT: AMBIGUOUS",
	])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is False
	assert verdict.category == "minor"


def test_escalation_error_fails_closed():
	reviewer, _ = make_reviewer([
		"VERDICT: SKIP_MINOR",
		RuntimeError("api down"),
	])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is False
	assert verdict.category == "minor"


def test_first_pass_error_fails_open():
	reviewer, _ = make_reviewer([RuntimeError("timeout")])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is True


def test_unparsable_verdict_approves_with_note():
	reviewer, _ = make_reviewer(["the model rambled without a verdict line"])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is True


def test_from_config_without_key_returns_none():
	assert OpenRouterReviewer.from_config({}) is None
	assert OpenRouterReviewer.from_config({"vision_review": True}) is None


def test_from_config_reads_models():
	cfg = {"openRouterToken": "k", "reviewer_model": "m1", "reviewer_escalation_model": "m2"}
	r = OpenRouterReviewer.from_config(cfg)
	assert r is not None
	assert r.first_pass_model == "m1"
	assert r.escalation_model == "m2"


def test_verdict_dataclass_has_category_default():
	v = Verdict(approved=True)
	assert v.category is None


# --- high-scrutiny confirmation (large-chest approvals get a second judge) ---

def test_chest_prompt_approval_requires_confirmation():
	reviewer, completions = make_reviewer([
		"VERDICT: GOOD",
		"confirmed clean\nVERDICT: GOOD",
	])
	verdict = run(reviewer.review(b"img", "woman, huge breasts, bedroom"))
	assert verdict.approved is True
	assert len(completions.calls) == 2  # first pass + confirmation
	assert "confirmed" in verdict.notes


def test_chest_prompt_confirmation_can_overturn_approval():
	reviewer, completions = make_reviewer([
		"VERDICT: GOOD",
		"breasts attach over the mid-abdomen\nVERDICT: BAD: torso elongated\nFIX: neg:extreme sizes",
	])
	verdict = run(reviewer.review(b"img", "woman, massive breasts, bedroom"))
	assert verdict.approved is False
	assert verdict.category == "quality"
	assert "mid-abdomen" in verdict.notes
	assert verdict.fix_negate == ["extreme sizes"]


def test_plain_prompt_approval_skips_confirmation():
	reviewer, completions = make_reviewer(["VERDICT: GOOD"])
	verdict = run(reviewer.review(b"img", "woman in a green dress, garden"))
	assert verdict.approved is True
	assert len(completions.calls) == 1


def test_confirmation_outage_fails_open():
	reviewer, _ = make_reviewer(["VERDICT: GOOD", RuntimeError("grok down")])
	verdict = run(reviewer.review(b"img", "woman, huge breasts, bedroom"))
	assert verdict.approved is True
	assert "confirmation unavailable" in verdict.notes


def test_confirmation_can_be_disabled():
	reviewer, completions = make_reviewer(["VERDICT: GOOD"])
	reviewer.confirm_high_scrutiny = False
	verdict = run(reviewer.review(b"img", "woman, huge breasts, bedroom"))
	assert verdict.approved is True
	assert len(completions.calls) == 1


def test_qc_prompt_carries_casebook_hints():
	from imagine_generator_proxy.review.openrouter import QC_PROMPT
	# the evolving failure-mode casebook from rule_data.json is injected
	assert "failure modes" in QC_PROMPT
	assert "fused or extra fingers" in QC_PROMPT


# --- session context (regeneration attempt chains) ---

def test_no_session_context_on_first_attempt():
	reviewer, completions = make_reviewer(["VERDICT: GOOD"])
	run(reviewer.review(b"img", "prompt", attempt=0, previous_notes=""))
	assert "SESSION CONTEXT" not in completions.prompts[0]


def test_session_context_injected_on_retry_attempt():
	reviewer, completions = make_reviewer(["VERDICT: GOOD"])
	run(reviewer.review(b"img", "prompt", attempt=1, previous_notes="fused fingers on left hand"))
	prompt = completions.prompts[0]
	assert "SESSION CONTEXT" in prompt
	assert "attempt 2" in prompt
	assert "fused fingers on left hand" in prompt


# --- FIX proposals ---

def test_parse_fix_extracts_add_and_negate():
	from imagine_generator_proxy.review.openrouter import parse_fix
	add, negate = parse_fix("VERDICT: BAD: melted torso\nFIX: arms at sides, sitting upright, neg:fused fingers")
	assert add == ["arms at sides", "sitting upright"]
	assert negate == ["fused fingers"]


def test_parse_fix_absent_returns_empty():
	from imagine_generator_proxy.review.openrouter import parse_fix
	assert parse_fix("VERDICT: GOOD") == ([], [])
	assert parse_fix("") == ([], [])


def test_parse_fix_caps_entries_and_drops_prose():
	from imagine_generator_proxy.review.openrouter import parse_fix
	long_prose = "a" * 80
	add, negate = parse_fix(
		f"FIX: t1, {long_prose}, t2, neg:n1, neg:n2, t3, t4, t5, t6, t7, t8")
	assert long_prose not in add
	assert len(add) + len(negate) <= 6


def test_bad_verdict_carries_fix_into_verdict():
	reviewer, _ = make_reviewer(["QC report\nVERDICT: BAD: extra arms\nFIX: arms at sides, neg:fused fingers"])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.approved is False
	assert verdict.fix_add == ["arms at sides"]
	assert verdict.fix_negate == ["fused fingers"]


def test_good_verdict_has_no_fix():
	reviewer, _ = make_reviewer(["VERDICT: GOOD"])
	verdict = run(reviewer.review(b"img", "prompt"))
	assert verdict.fix_add == [] and verdict.fix_negate == []


# --- retrieval hints (review history) ---

def test_history_injected_when_available():
	import sqlite3, tempfile, os
	from imagine_generator_proxy.server.storage import ensure_db_schema

	fd, path = tempfile.mkstemp(suffix=".db")
	os.close(fd)
	conn = sqlite3.connect(path)
	ensure_db_schema(conn)
	conn.execute("INSERT INTO reviews (digest, character, approved, category, notes, attempt) "
		"VALUES ('deadbeef', 'Naimi', 0, 'quality', 'melted torso on large chest', 0)")
	conn.commit()
	conn.close()

	reviewer, completions = make_reviewer(["VERDICT: GOOD"])
	reviewer._db_path = path
	run(reviewer.review(b"img", "prompt", character="Naimi"))
	os.unlink(path)
	assert "REVIEW HISTORY" in completions.prompts[0]
	assert "melted torso" in completions.prompts[0]


def test_no_history_block_without_db():
	reviewer, completions = make_reviewer(["VERDICT: GOOD"])
	run(reviewer.review(b"img", "prompt", character="Naimi"))
	assert "REVIEW HISTORY" not in completions.prompts[0]
