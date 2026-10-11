"""Breast-size tag ladder: which tags (and emphasis) actually scale up?

Generates each variant through the production proxy path (massager + NovelAI),
then measures the result with qwen3-vl (temperature 0). 2 samples per variant
because generation is unseeded — a single sample is noise.

Usage: python scripts/size_ladder.py
Writes images + results.json under data/experiments/size_ladder/
"""

from __future__ import annotations

import base64
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "experiments" / "size_ladder"
PROXY = "http://localhost:8080/generate"

BASE = ("1girl, solo, young woman, extremely long blonde hair, blue eyes, "
	"standing, front view, full body, simple background")

# variant -> (size tags appended, negative prompt or "")
VARIANTS = {
	"A_large": ("large breasts", ""),
	"B_huge": ("huge breasts", ""),
	"C_huge_emph": ("{{huge breasts}}", ""),
	"D_massive": ("massive breasts", ""),
	"E_massive_heavy": ("massive heavy breasts", ""),
	"F_gigantic": ("gigantic breasts", ""),
	"G_huge_headcue": ("huge breasts, breasts bigger than her head", ""),
	"H_huge_negup": ("huge breasts", "medium breasts, large breasts"),
}

REPS = 2

MEASURE_PROMPT = """Measure this anime figure's breast size. Answer in EXACTLY three lines:
SIZE: <one word: flat | small | medium | large | huge | enormous | gigantic>
HEAD: <are the breasts larger than her head: yes | no>
TORSO: <is the torso stretched, distended, or anatomically broken to hold the chest: yes | no>"""


def generate(prompt: str, negative: str) -> bytes:
	url = PROXY + "?" + urllib.parse.urlencode({
		"prompt": prompt, "negative_prompt": negative,
		"raw": "1", "force": "1"})  # raw bytes, bypass cache for fresh samples
	with urllib.request.urlopen(url, timeout=180) as resp:
		return resp.read()


def measure(api_key: str, image: bytes) -> dict:
	body = json.dumps({"model": "qwen/qwen3-vl-32b-instruct", "max_tokens": 60,
		"temperature": 0.0,
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": MEASURE_PROMPT},
			{"type": "image_url", "image_url": {
				"url": f"data:image/png;base64,{base64.b64encode(image).decode()}"}},
		]}]}).encode()
	req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
		data=body, headers={"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json"})
	with urllib.request.urlopen(req, timeout=90) as resp:
		text = json.load(resp)["choices"][0]["message"]["content"].strip().lower()
	size = next((w for w in ("flat", "small", "medium", "large", "huge",
			"enormous", "gigantic") if w in text.split()), "?")
	return {"size": size,
		"head": "yes" if "head: yes" in text else "no",
		"torso": "yes" if "torso: yes" in text else "no"}


def main() -> None:
	config = yaml.safe_load(open(ROOT / "data" / "config.yaml", encoding="utf-8"))
	key = config["openRouterToken"]
	OUT.mkdir(parents=True, exist_ok=True)
	results = {}
	for name, (tags, neg) in VARIANTS.items():
		prompt = f"{BASE}, {tags}"
		for rep in range(1, REPS + 1):
			try:
				img = generate(prompt, neg)
				path = OUT / f"{name}_{rep}.png"
				path.write_bytes(img)
				m = measure(key, img)
				results[f"{name}_{rep}"] = {"variant": name, "rep": rep,
					"tags": tags, "negative": neg, **m}
				print(f"{name} r{rep}: size={m['size']} head={m['head']} torso={m['torso']}",
					flush=True)
			except Exception as ex:  # noqa: BLE001 — keep the ladder alive
				results[f"{name}_{rep}"] = {"variant": name, "rep": rep, "error": str(ex)}
				print(f"{name} r{rep}: FAILED {ex}", flush=True)
			time.sleep(2)
	with open(OUT / "results.json", "w", encoding="utf-8") as fh:
		json.dump(results, fh, indent=2)
	print("saved", OUT / "results.json")


if __name__ == "__main__":
	main()
