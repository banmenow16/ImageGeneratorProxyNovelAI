"""Standalone vision MCP: honest facts from vision models, no strings.

Independent of the image proxy server — reads the OpenRouter key from
the proxy's config (data/config.yaml) but talks to OpenRouter directly,
so benches and QC keep working through image-server restarts and model
swaps. Register in .mcp.json as its own server; any Claude session on
the machine can use it.

The prompting laws this tool exists to enforce (learned the hard way,
2026-10-09/10):
- NEUTRAL FACTS + attention cue, temperature 0. A contract-primed
  prompt makes the model see what it's told to.
- Continuous quantities get PERCENTAGE estimates, never ordered enums —
  leading lists anchor the judge to the first option (15/16 identical
  answers measured).
- Pairwise comparison must be ORDER-SWAPPED; compare_images does both
  orders and reports each.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("VISION_CONFIG")
	or ROOT / "data" / "config.yaml")
DEFAULT_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3-vl-235b-a22b-instruct")
MAX_IMAGES = 4


def load_api_key() -> str:
	"""OpenRouter key from the proxy config (YAML or flat key:value)."""
	text = CONFIG_PATH.read_text(encoding="utf-8")
	match = re.search(r"(?:^|\n)\s*openRouterToken:\s*[\"']?([^\"'\n]+)", text)
	if not match:
		raise SystemExit(f"no openRouterToken found in {CONFIG_PATH}")
	return match.group(1).strip()


API_KEY = load_api_key()


class OpenRouterVision:
	"""Thin async client: images + prompt -> model text."""

	def __init__(self, api_key: str, timeout_s: float = 180.0) -> None:
		self._api_key = api_key
		self._timeout = aiohttp.ClientTimeout(total=timeout_s)

	async def ask(self, images: list[str], prompt: str, model: str,
			temperature: float, max_tokens: int) -> dict:
		model = model or DEFAULT_MODEL
		content: list[dict] = [{"type": "text", "text": prompt}]
		for image in images:
			content.append({"type": "image_url", "image_url":
				{"url": self._image_url(image)}})
		body = {"model": model, "messages": [{"role": "user", "content": content}],
			"temperature": temperature, "max_tokens": max_tokens}
		try:
			async with aiohttp.ClientSession(timeout=self._timeout) as session:
				async with session.post("https://openrouter.ai/api/v1/chat/completions",
						headers={"Authorization": f"Bearer {self._api_key}",
							"Content-Type": "application/json"},
						json=body) as resp:
					data: dict = await resp.json(content_type=None)
					if resp.status >= 400:
						return {"error": f"openrouter {resp.status}: "
							+ str(data)[:400]}
					return {"text": (data.get("choices") or [{}])[0]
						.get("message", {}).get("content", "") or "",
						"model": model}
		except aiohttp.ClientError as ex:
			return {"error": f"request failed: {ex}"}

	@staticmethod
	def _image_url(image: str) -> str:
		"""Local path -> data URL; URLs and data: URLs pass through."""
		if image.startswith(("http://", "https://", "data:")):
			return image
		path = Path(image)
		payload = base64.b64encode(path.read_bytes()).decode("ascii")
		return f"data:image/png;base64,{payload}"


client = OpenRouterVision(API_KEY)

try:
	from fastmcp import FastMCP
except ImportError as ex:  # pragma: no cover — surfaced to the operator
	raise SystemExit("fastmcp is required: pip install fastmcp") from ex

mcp = FastMCP("vision")


@mcp.tool
async def analyze_images(images: list[str], prompt: str, model: str = "",
		temperature: float = 0.0, max_tokens: int = 400) -> dict:
	"""Ask a vision model about 1-4 images and get raw text back.
	Images are local file paths, http(s) URLs, or data: URLs. For
	reliable answers follow the prompting laws: ask for neutral FACTS
	with an explicit attention cue ("look closely at X"), temperature 0,
	and for any continuous quantity ask for a PERCENTAGE estimate —
	never an ordered multiple-choice list (leading enums anchor the
	model to the first option)."""
	return await client.ask(images, prompt, model or DEFAULT_MODEL,
		temperature, max_tokens)


@mcp.tool
async def compare_images(image_a: str, image_b: str, question: str,
		model: str = "", max_tokens: int = 90) -> dict:
	"""Pairwise comparison, both orders, anchor-resistant. Ask an
	open question like 'which image shows the larger breasts' or
	'which is higher quality'. The tool asks it with A first, then B
	first, and reports both answers plus a verdict: 'A', 'B', or
	'INCONSISTENT' (position bias — treat as no difference)."""
	q1 = (f"Two images, IMAGE 1 first then IMAGE 2. {question.replace('which', 'Which', 1)} "
		"Answer in exactly one line: ANSWER: <IMAGE 1 or IMAGE 2 or EQUAL> (<one short reason>)")
	first = await client.ask([image_a, image_b], q1, model or DEFAULT_MODEL,
		0.0, max_tokens)
	second = await client.ask([image_b, image_a], q1, model or DEFAULT_MODEL,
		0.0, max_tokens)

	def pick(text: str) -> str:
		if "IMAGE 1" in text: return "first-listed"
		if "IMAGE 2" in text: return "second-listed"
		return "equal"

	p1, p2 = pick(first.get("text", "")), pick(second.get("text", ""))
	verdict = ("A" if p1 == "first-listed" and p2 == "second-listed"
		else "B" if p1 == "second-listed" and p2 == "first-listed"
		else "equal" if p1 == "equal" and p2 == "equal" else "INCONSISTENT")
	return {"verdict": verdict, "a_first": first, "b_first": second,
		"note": "INCONSISTENT means position bias won — treat as no "
			"measurable difference"}


if __name__ == "__main__":
	mcp.run()
