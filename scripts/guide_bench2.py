"""Bench 2: emphasis syntax, numeric weights, beyond-enormous sizes,
nipple recipes, framing ladder, bracket weakening, garment colors.

Second measured round for data/image_prompt_guide.md — mixes official-
tutorial techniques ([tag] weakening, N::tag:: weights, framing ladder,
colored garments) with the open questions from bench 1. Resumable: gen
records keyed by label; reruns skip paid generations and re-judge.

Writes data/experiments/guide_claims_20261009/bench2.ndjson.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "experiments" / "guide_claims_20261009" / "bench2.ndjson"
GEN = "http://localhost:8080/api/generate"
VISION = "http://localhost:8080/api/vision/analyze"

IDENTITY = ("1girl, solo, young woman, shoulder-length black hair, amber eyes, "
	"standing, front view, simple background")
ID_NUDE = IDENTITY + ", completely nude, full body"

records: list[dict] = ([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()
	if l.strip()] if OUT.exists() else [])


def post(url: str, payload: dict, timeout: int = 180) -> dict:
	body = json.dumps(payload).encode()
	req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
	return json.load(urllib.request.urlopen(req, timeout=timeout))


def log(record: dict) -> None:
	records.append(record)
	OUT.open("a", encoding="utf-8").write(json.dumps(record) + "\n")


def generate(label: str, prompt: str, negative: str = "") -> str:
	for rec in records:
		if rec.get("kind") == "gen" and rec.get("label") == label:
			return rec["path"]
	resp = post(GEN, {"prompt": prompt, "negative": negative, "save": True})
	time.sleep(1.3)
	path = resp.get("path", "")
	log({"kind": "gen", "label": label, "path": path,
		"massaged": resp.get("massaged_prompt", "")[:500]})
	print(f"[gen] {label} -> {path}", flush=True)
	return path


def vision(prompt: str, images: list[str], max_tokens: int = 220) -> str:
	try:
		resp = post(VISION, {"images": images, "prompt": prompt,
			"temperature": 0.0, "max_tokens": max_tokens}, timeout=120)
		return resp.get("text", "").strip()
	except Exception as ex:  # noqa: BLE001 — judge failures are data too
		return f"JUDGE-ERROR: {ex}"


def judge(label: str, prompt: str, image: str, max_tokens: int = 220) -> str:
	text = vision(prompt, [image], max_tokens)
	log({"kind": "judge", "label": label, "path": image, "text": text[:600]})
	print(f"[judge] {label}: {text[:110]}", flush=True)
	return text


SIZE_COMPARE = ("Two images of one woman, IMAGE 1 first then IMAGE 2. Look "
	"closely at both busts. Which shows the larger breasts, IMAGE 1 or "
	"IMAGE 2? Answer exactly: LARGER: <IMAGE 1 or IMAGE 2 or EQUAL> (<reason>)")


def pairwise(label_a: str, paths_a: list[str], label_b: str,
		paths_b: list[str]) -> dict:
	wins = {label_a: 0, label_b: 0, "EQUAL": 0}
	for pa in paths_a:
		for pb in paths_b:
			for first, second, w1, w2 in ((pa, pb, label_a, label_b),
					(pb, pa, label_b, label_a)):
				ans = vision(SIZE_COMPARE, [first, second], 90)
				winner = (w1 if "IMAGE 1" in ans else
					w2 if "IMAGE 2" in ans else "EQUAL")
				wins[winner] += 1
				log({"kind": "pairwise", "a": w1, "b": w2, "winner": winner,
					"ans": ans[:120]})
				print(f"[pair] {w1} vs {w2}: {winner}", flush=True)
	print(f"[pair-summary] {label_a} vs {label_b}: {wins}", flush=True)
	return wins


def main() -> None:
	# --- G: emphasis A/B on the gallery identity (fixes the bad renders) ---
	g_plain = [generate(f"G_plain_s{s}", f"{ID_NUDE}, huge breasts")
		for s in (1, 2)]
	g_braced = [generate(f"G_braced_s{s}", f"{ID_NUDE}, {{{{huge breasts}}}}")
		for s in (1, 2)]
	pairwise("G_plain_huge", g_plain, "G_braced_huge", g_braced)

	# --- E: beyond enormous (control = gallery enormous_1/2) ---------------
	enormous = ["debug/guide_gallery/enormous_1.png",
		"debug/guide_gallery/enormous_2.png"]
	e_variants = {
		"E_hyper": f"{ID_NUDE}, hyper breasts",
		"E_gigantic": f"{ID_NUDE}, gigantic breasts",
		"E_w13": f"{ID_NUDE}, 1.3::enormous breasts::",
		"E_w15": f"{ID_NUDE}, 1.5::enormous breasts::",
	}
	for label, prompt in e_variants.items():
		paths = [generate(f"{label}_s{s}", prompt) for s in (1, 2)]
		pairwise("E_enormous", enormous, label, paths)

	# --- F: nipple recipes (topless) ---------------------------------------
	f_base = (IDENTITY + ", large breasts, topless, bare breasts, upper body")
	f_variants = {
		"F_control": f_base + ", nipples",
		"F_puffy": f_base.replace(", nipples", ", puffy erect nipples"),
		"F_swollen": f_base.replace(", nipples", ", swollen nipples"),
		"F_small_areola": f_base.replace(", nipples", ", small areola"),
		"F_textured": f_base + ", detailed textured areola",
	}
	NIP = ("Study the chest closely. Answer exactly two lines: "
		"NIPPLES: <drawn or absent> "
		"STYLE: <for each: size, color, erect/flat, areola detail>")
	for label, prompt in f_variants.items():
		for s in (1, 2):
			path = generate(f"{label}_s{s}", prompt)
			judge(f"{label}_s{s}", NIP, path)

	# --- A: bracket weakening (hair messiness) ------------------------------
	a_base = IDENTITY + ", full body"
	a_variants = {
		"A_none": a_base,
		"A_plain": a_base + ", messy hair",
		"A_braced": a_base + ", {{{messy hair}}}",
		"A_bracketed": a_base + ", [[[messy hair]]]",
	}
	MESS = ("Look closely at the hair. Rate messiness 0-10 (0 perfectly "
		"neat and combed, 10 extremely wild and tangled). Answer exactly: "
		"MESS: <n>/10 - <one short description>")
	for label, prompt in a_variants.items():
		for s in (1, 2):
			path = generate(f"{label}_s{s}", prompt)
			judge(f"{label}_s{s}", MESS, path, 120)

	# --- B: numeric weight on silver (drift fix candidate) ------------------
	b_base = (IDENTITY + ", full body, silver hair, tiara")
	b_plain = [generate(f"B_plain_s{s}", b_base) for s in (1, 2)]
	b_weight = [generate(f"B_weight_s{s}",
		b_base.replace("silver hair", "1.5::silver hair::")) for s in (1, 2)]
	COLOR = ("Look closely. Answer exactly: HAIR: <color>, TIARA: <color or none>")
	for label, paths in (("B_plain", b_plain), ("B_weight", b_weight)):
		for s, path in zip((1, 2), paths):
			judge(f"{label}_s{s}", COLOR, path, 90)

	# --- C: framing ladder ---------------------------------------------------
	c_variants = {"C_full": "full body", "C_cowboy": "cowboy shot",
		"C_wide": "wide shot"}
	FRAME = ("Report the framing. Answer exactly: FRAME: <lowest visible body "
		"part: feet / ankles / knees / thighs / hips / waist / chest> - "
		"<figure size in frame: fills / half / small>")
	for label, framing in c_variants.items():
		for s in (1, 2):
			path = generate(f"{label}_s{s}",
				f"{IDENTITY}, {framing}, white shirt, blue skirt")
			judge(f"{label}_s{s}", FRAME, path, 120)

	# --- D: garment color consistency ---------------------------------------
	d_variants = {
		"D_uncolored": IDENTITY + ", full body, shirt, skirt",
		"D_colored": IDENTITY + ", full body, white shirt, blue skirt",
	}
	GAR = ("Answer exactly: SHIRT: <color or none>, SKIRT: <color or none>")
	for label, prompt in d_variants.items():
		for s in (1, 2, 3):
			path = generate(f"{label}_s{s}", prompt)
			judge(f"{label}_s{s}", GAR, path, 90)

	print("BENCH2 DONE", flush=True)


if __name__ == "__main__":
	main()
