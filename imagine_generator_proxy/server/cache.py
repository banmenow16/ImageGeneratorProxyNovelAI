"""Cache management: reading/writing cached images and preparing the HTTP
responses that carry them (raw PNG or JSON base64), including the
effective-prompt transparency headers/payloads."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from aiohttp import web

LOGGER = logging.getLogger(__name__)


def response_headers(effective_prompt: Optional[str]) -> dict:
	"""Headers exposing the effective prompt on raw (PNG) responses."""
	if effective_prompt is None:
		return {}
	return {"X-Effective-Prompt": quote(effective_prompt, safe="")}


def _payload_with_prompt(payload: dict, effective_prompt: Optional[str],
		massager_version: Optional[int]) -> dict:
	if effective_prompt is not None:
		payload["effective_prompt"] = effective_prompt
		payload["massager_version"] = massager_version
		payload["massaged"] = massager_version is not None
	return payload


def read_cached_image(cached_file: Path, want_raw: bool, context: str = None,
		effective_prompt: Optional[str] = None, massager_version: Optional[int] = None):
	"""Return a web Response or JSON response for the cached file, or None if not available.

	`context` is used only to customize log messages when read fails.
	"""
	try:
		if not cached_file.exists():
			return None

		data = cached_file.read_bytes()
		if want_raw:
			return web.Response(body=bytes(data), content_type="image/png",
				headers=response_headers(effective_prompt))

		payload = _payload_with_prompt(
			{"name": cached_file.name, "image_b64": base64.b64encode(data).decode("utf-8"), "cached": True},
			effective_prompt, massager_version)
		return web.json_response(payload)
	except Exception:
		if context:
			LOGGER.exception("Failed to read cached image (%s)", context)
		else:
			LOGGER.exception("Failed to read cached image")
		return None


def write_cache_and_prepare_response(data, char_dir: Path, cached_file: Path, want_raw: bool,
		gens: dict, digest: str, name: str,
		effective_prompt: Optional[str] = None, massager_version: Optional[int] = None):
	try:
		char_dir.mkdir(parents=True, exist_ok=True)
		cached_file.write_bytes(bytes(data))
	except Exception:
		LOGGER.exception("Failed to write cache file")

	# signal waiting requests before returning
	try:
		gens[digest].set()
	except Exception:
		pass
	gens.pop(digest, None)

	if want_raw:
		return web.Response(body=bytes(data), content_type="image/png",
			headers=response_headers(effective_prompt))

	payload = _payload_with_prompt(
		{"name": name, "image_b64": base64.b64encode(data).decode("utf-8"), "cached": False},
		effective_prompt, massager_version)
	return web.json_response(payload)


def prepare_nonbinary_response(name: str, data, gens: dict, digest: str):
	try:
		gens[digest].set()
	except Exception:
		pass
	gens.pop(digest, None)
	return web.json_response({"name": name, "image": str(data)})
