"""Unit tests for the two-tier OpenRouter reviewer (stubbed client)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from imagine_generator_proxy.openrouter_reviewer import (
	OpenRouterReviewer,
	parse_assessment,
	parse_verdict,
)
from imagine_generator_proxy.reviewer import Verdict


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


def test_qc_prompt_carries_casebook_hints():
	from imagine_generator_proxy.openrouter_reviewer import QC_PROMPT
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
