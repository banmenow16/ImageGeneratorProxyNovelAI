"""Bench 3: nipple size/shape control in a close-up context.

Bench 2 showed style recipes inert in a neutral standing render; the
open question is whether the dials work where nipple detail actually has
pixels - a torso close-up. Ten variants (size, puffiness, areola,
inversion, erection), two samples each, judged with a per-nipple fact
read at temperature 0.

Writes data/experiments/guide_claims_20261009/bench3.ndjson.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "experiments" / "guide_claims_20261009" / "bench3.ndjson"
GEN = "http://localhost:8080/api/generate"
VISION = "http://localhost:8080/api/vision/analyze"

BASE = ("1girl, solo, young woman, shoulder-length black hair, large breasts, "
	"topless, bare breasts, close-up view, torso framing, looking at viewer, "
	"bedroom background blurred")
VARIANTS = {
	"N_control": "",
	"N_big": ", big nipples",
	"N_huge": ", huge nipples",
	"N_long": ", long nipples",
	"N_puffy": ", puffy nipples",
	"N_puffy_large": ", puffy nipples, large areola",
	"N_large_areola": ", large areola",
	"N_small_areola": ", small areola",
	"N_inverted": ", inverted nipples",
	"N_erect": ", erect nipples",
}
JUDGE = ("Look closely at each nipple and areola in this image. Answer in "
	"exactly three lines:\n"
	"NIPPLES: <drawn or absent>\n"
	"SHAPE: <one of flat / erect-protruding / puffy-raised-cushion / "
	"long-elongated / inverted-sunken, or one per side if they differ>\n"
	"SIZE+AREOLA: <nipple size tiny/small/medium/large/huge; areola "
	"small/medium/large; surface smooth or bumpy-textured>")

records: list = ([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()
	if l.strip()] if OUT.exists() else [])


def post(url, payload, timeout=180):
	body = json.dumps(payload).encode()
	req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
	return json.load(urllib.request.urlopen(req, timeout=timeout))


def log(record):
	records.append(record)
	OUT.open("a", encoding="utf-8").write(json.dumps(record) + "\n")


def generate(label, prompt):
	for rec in records:
		if rec.get("kind") == "gen" and rec.get("label") == label:
			return rec["path"]
	resp = post(GEN, {"prompt": prompt, "negative": "", "save": True})
	time.sleep(1.3)
	path = resp.get("path", "")
	log({"kind": "gen", "label": label, "path": path,
		"massaged": resp.get("massaged_prompt", "")[:400]})
	print("[gen] %s -> %s" % (label, path), flush=True)
	return path


def main():
	for label, extra in VARIANTS.items():
		for s in (1, 2):
			label_s = "%s_s%s" % (label, s)
			path = generate(label_s, BASE + extra)
			for attempt in (1, 2, 3):
				try:
					resp = post(VISION, {"images": [path], "prompt": JUDGE,
						"temperature": 0.0, "max_tokens": 200}, timeout=120)
					text = resp.get("text", "").strip()
					break
				except Exception as ex:
					text = "JUDGE-ERROR: %s" % ex
					time.sleep(8)
			log({"kind": "judge", "label": label_s, "path": path,
				"text": text[:600]})
			print("[judge] %s: %s" % (label_s, text.replace(chr(10), " | ")[:130]),
				flush=True)
	print("BENCH3 DONE", flush=True)


if __name__ == "__main__":
	main()
