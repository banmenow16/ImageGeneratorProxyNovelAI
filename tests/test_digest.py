"""Digest tests: the legacy formula must stay byte-identical, the versioned
formula must namespace massaged prompts and be stable."""

from __future__ import annotations

import hashlib

from novelai_api.ImagePreset import ImageModel

from imagine_generator_proxy.storage import compute_digest

CHARACTER = "Naimi"
PROMPT = "score_9, score_8_up, source_anime,red dress"
NEGATIVE = "text, watermark, deformed, ugly, "
MODEL = ImageModel.Anime_v45_Full


def test_legacy_digest_matches_historical_formula():
	# the exact formula that shipped before the massager existed
	key_material = "|".join([CHARACTER, PROMPT, NEGATIVE, str(MODEL)])
	expected = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
	assert compute_digest(CHARACTER, PROMPT, NEGATIVE, MODEL, None) == expected


def test_versioned_digest_differs_from_legacy():
	legacy = compute_digest(CHARACTER, PROMPT, NEGATIVE, MODEL, None)
	versioned = compute_digest(CHARACTER, PROMPT, NEGATIVE, MODEL, 1)
	assert legacy != versioned


def test_versioned_digest_is_stable_and_namespaced():
	v1_first = compute_digest(CHARACTER, PROMPT, NEGATIVE, MODEL, 1)
	v1_again = compute_digest(CHARACTER, PROMPT, NEGATIVE, MODEL, 1)
	v2 = compute_digest(CHARACTER, PROMPT, NEGATIVE, MODEL, 2)
	assert v1_first == v1_again
	assert v1_first != v2
