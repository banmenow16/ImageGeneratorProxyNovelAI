"""Build the guide-evidence gallery page from experiment records.

Generates the ladder rungs that have no images yet (flat/small/medium/
large, nude, one fixed identity), copies every referenced image into
data/debug/guide_gallery/ under clean names, and writes gallery.html —
served at /debug-files/guide_gallery/ and openable as a local file.

Resumable: rung generations are tracked in rungs.json inside the gallery
folder; existing records from data/experiments/guide_claims_20261009/
provide the huge/massive/enormous, cue, two-girl, and bra images.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / "data" / "debug" / "guide_gallery"
EXPERIMENTS = ROOT / "data" / "experiments" / "guide_claims_20261009"
OLD_LADDER = ROOT / "data" / "experiments" / "size_ladder"
GEN = "http://localhost:8080/api/generate"

IDENTITY = ("1girl, solo, young woman, shoulder-length black hair, amber eyes, "
	"completely nude, standing, front view, full body, simple background, ")
RUNGS = {"flat": "flat chest", "small": "small breasts",
	"medium": "medium breasts", "large": "large breasts"}


def generate(tag: str, rep: int, records: dict) -> str:
	key = f"{tag}_{rep}"
	if key in records:
		return records[key]["path"]
	body = json.dumps({"prompt": IDENTITY + tag, "negative": "",
		"save": True}).encode()
	req = urllib.request.Request(GEN, body, {"Content-Type": "application/json"})
	resp = json.load(urllib.request.urlopen(req, timeout=300))
	records[key] = {"path": resp["path"], "tag": tag}
	time.sleep(1.3)
	return resp["path"]


def load_ndjson(path: Path) -> list[dict]:
	return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
		if l.strip()]


def main() -> None:
	GALLERY.mkdir(parents=True, exist_ok=True)
	state_file = GALLERY / "rungs.json"
	records: dict = (json.loads(state_file.read_text(encoding="utf-8"))
		if state_file.exists() else {})

	# 1. generate missing lower rungs (paid, resumable)
	for tag in RUNGS.values():
		for rep in (1, 2):
			path = generate(tag, rep, records)
			print("[gen]", tag, rep, "->", path, flush=True)
	state_file.write_text(json.dumps(records, indent=1), encoding="utf-8")

	# 2. collect existing evidence from the experiment records
	ladder = {r["label"]: r["path"] for r in
		load_ndjson(EXPERIMENTS / "ladder_replication.ndjson")
		if r.get("kind") == "gen"}
	claims = {r["label"]: r["path"] for r in
		load_ndjson(EXPERIMENTS / "guide_experiments_results.ndjson")
		if r.get("kind") == "gen"}
	bench2_events = load_ndjson(EXPERIMENTS / "bench2.ndjson")
	bench2 = {r["label"]: r["path"] for r in bench2_events
		if r.get("kind") == "gen"}
	bench2_judged = {r["label"]: r.get("text", "") for r in bench2_events
		if r.get("kind") == "judge"}

	def pair_wins(a: str, b: str) -> dict:
		wins = {a: 0, b: 0, "EQUAL": 0}
		for r in bench2_events:
			if r.get("kind") == "pairwise" and {r["a"], r["b"]} == {a, b}:
				wins[r["winner"]] = wins.get(r["winner"], 0) + 1
		return wins

	# 3. copy into the gallery under clean names
	def copy(src_name: str, src_path: str) -> str:
		src = ROOT / "data" / src_path
		dst = GALLERY / src_name
		if src.is_file():
			shutil.copyfile(src, dst)
		return src_name

	def rung_images(tag: str, label: str) -> list[str]:
		names = []
		for rep in (1, 2):
			key = f"{tag}_{rep}"
			if key in records:
				names.append(copy(f"{label}_{rep}.png", records[key]["path"]))
		return names

	sections: list[dict] = []

	# --- the ladder -------------------------------------------------------
	rows = []
	for label, tag, note in (
			("flat", "flat chest", "default-up risk: `small breasts` renders medium — use flat/petite when flat matters"),
			("small", "small breasts", "renders a notch above label; expect medium"),
			("medium", "medium breasts", "verify on first image and tag down if needed"),
			("large", "large breasts", "can render head-exceeding on busty-leaning prompts"),
			("huge", "huge breasts", "beaten by enormous 15:1 combined (clothed 8:0, nude 7:1)"),
			("massive", "massive breasts", "> huge nude (5:3); `massive heavy breasts` is WEAKER (2:6) — never add the word"),
			("enormous", "enormous breasts", "the strongest measured tag — beat huge and massive in both contexts")):
		if label in ("huge", "massive", "enormous"):
			imgs = [copy(f"{label}_{rep}.png", ladder[f"{label}_s{rep}"])
				for rep in (1, 2)]
		else:
			imgs = rung_images(RUNGS[label], label)
		rows.append({"tag": tag, "note": note, "images": imgs})
	sections.append({"title": "Breast size ladder (one identity, nude)",
		"intro": "Vocabulary controls size — braces don't. One tag per character, "
			"in the identity block. Pairwise-verified, order-swapped vision "
			"comparisons; evidence in data/experiments/guide_claims_20261009/.",
		"rows": rows})

	# --- emphasis doesn't scale (bench 2, same identity as the ladder) -----
	emph_rows = [
		{"tag": "huge breasts", "note": "plain control",
			"images": [copy(f"emph_huge_{rep}.png", bench2[f"G_plain_s{rep}"])
				for rep in (1, 2)]},
		{"tag": "{{huge breasts}}", "note": "braced — same size or smaller, "
			"never bigger (pairwise 4:4, a wash)",
			"images": [copy(f"emph_braced_{rep}.png", bench2[f"G_braced_s{rep}"])
				for rep in (1, 2)]},
	]
	sections.append({"title": "Emphasis does not scale",
		"intro": "`{{huge breasts}}` rendered the same as plain `huge` "
			"(bench 2, order-swapped pairwise: 4:4). Braces are for ignored "
			"features, not sizing.",
		"rows": emph_rows})

	# --- beyond enormous ----------------------------------------------------
	e_rows = [{"tag": "enormous breasts (control)", "note": "the ceiling",
		"images": ["enormous_1.png", "enormous_2.png"]}]
	e_specs = (
		("E_hyper", "hyper breasts", "ties enormous"),
		("E_gigantic", "gigantic breasts", "ties enormous (and rendered "
			"below huge in bench 1 — unstable word)"),
		("E_w13", "1.3::enormous breasts::", "ties plain enormous"),
		("E_w15", "1.5::enormous breasts::", "ties plain enormous"))
	for label, tag, verdict in e_specs:
		wins = pair_wins("E_enormous", label)
		e_rows.append({"tag": tag,
			"note": f"{verdict} — pairwise {wins.get('E_enormous', 0)}:"
				f"{wins.get(label, 0)}",
			"images": [copy(f"{label}_{rep}.png", bench2[f"{label}_s{rep}"])
				for rep in (1, 2)]})
	sections.append({"title": "Nothing renders bigger than enormous",
		"intro": "Every candidate for going past `enormous breasts` landed "
			"within judge resolution of it (bench 2). Enormous is the "
			"ceiling; past it only torso distortion, which is a defect.",
		"rows": e_rows})

	# --- numeric weights hold colors ---------------------------------------
	b_rows = []
	for label, tag, note in (
			("B_plain", "silver hair, tiara", "hair drifted gray once; tiara gold both"),
			("B_weight", "1.5::silver hair::, tiara", "hair silver 2/2 — the weight holds")):
		b_rows.append({"tag": tag, "note": note,
			"images": [copy(f"{label}_{rep}.png", bench2[f"{label}_s{rep}"])
				for rep in (1, 2)]})
	sections.append({"title": "Numeric weights hold stubborn colors",
		"intro": "`N::tag::` is the strongest hold, above braces. The weight "
			"protects only the tag it wraps — the unweighted tiara kept "
			"drifting gold.",
		"rows": b_rows})

	# --- framing ladder -----------------------------------------------------
	c_rows = []
	for label, tag in (("C_full", "full body"), ("C_cowboy", "cowboy shot"),
			("C_wide", "wide shot")):
		reads = [bench2_judged.get(f"{label}_s{s}", "")[:60] for s in (1, 2)]
		c_rows.append({"tag": tag, "note": " | ".join(reads),
			"images": [copy(f"{label}_{rep}.png", bench2[f"{label}_s{rep}"])
				for rep in (1, 2)]})
	sections.append({"title": "Framing ladder",
		"intro": "`full body` reliable (feet 2/2, front view); `cowboy shot` "
			"renders waist-tight, tighter than the danbooru mid-thigh "
			"convention; `wide shot` pulls back to half-figure.",
		"rows": c_rows})

	# --- brackets weaken? (no measurable effect) -----------------------------
	a_rows = []
	for label, tag in (("A_none", "(no mess tag)"), ("A_plain", "messy hair"),
			("A_braced", "{{{messy hair}}}"), ("A_bracketed", "[[[messy hair]]]")):
		reads = [bench2_judged.get(f"{label}_s{s}", "")[:40] for s in (1, 2)]
		a_rows.append({"tag": tag, "note": " | ".join(reads),
			"images": [copy(f"{label}_{rep}.png", bench2[f"{label}_s{rep}"])
				for rep in (1, 2)]})
	sections.append({"title": "Square brackets show no measurable weakening",
		"intro": "Officially `[tag]` decreases emphasis — but `[[[messy hair]]]` "
			"rendered the same as plain (messiness reads 6/4 vs 6/6). Don't "
			"rely on brackets for tuning.",
		"rows": a_rows})

	# --- nipple dials (bench 3, close-up) ------------------------------------
	bench3_events = load_ndjson(EXPERIMENTS / "bench3.ndjson")
	bench3 = {r["label"]: r["path"] for r in bench3_events if r.get("kind") == "gen"}
	bench3_judged = {r["label"]: r.get("text", "") for r in bench3_events
		if r.get("kind") == "judge"}
	f_rows = []
	for label, tag in (("N_control", "(no dial — close-up control)"),
			("N_long", "long nipples"), ("N_big", "big nipples"),
			("N_huge", "huge nipples"),
			("N_w_long_big", "long nipples, 1.5::big nipples::"),
			("N_puffy", "puffy nipples"), ("N_inverted", "inverted nipples"),
			("N_w_inverted", "1.5::inverted nipples::"),
			("N_erect", "erect nipples"),
			("N_large_areola", "large areola"),
			("N_w_large_areola", "1.5::large areola::"),
			("N_swollen", "swollen nipples"),
			("N_mont", "glands of montgomery"),
			("N_w_mont", "1.5::glands of montgomery::"),
			("N_large_areolae", "large areolae (plural)")):
		reads = [bench3_judged.get(f"{label}_s{s}", "")[:55].replace("\n", " ")
			for s in (1, 2)]
		f_rows.append({"tag": tag, "note": " | ".join(reads),
			"images": [copy(f"{label}_{rep}.png", bench3[f"{label}_s{rep}"])
				for rep in (1, 2)]})
	sections.append({"title": "Nipple dials: long is the only dial that turns",
		"intro": "Torso close-up, n=2 per variant (bench 3; neutral-standing "
			"recipes were already inert in bench 2). `long nipples` reliably "
			"renders medium + protruding; weighting `big nipples` alongside "
			"pushes occasional large. Inverted, puffy, areola size, and bump "
			"texture: no dial found — inert even at 1.5:: weight. The cue "
			"controls WHETHER, almost nothing controls HOW.",
		"rows": f_rows})

	# --- garment colors ------------------------------------------------------
	d_rows = []
	for label, tag in (("D_uncolored", "shirt, skirt"),
			("D_colored", "white shirt, blue skirt")):
		reads = [bench2_judged.get(f"{label}_s{s}", "")[:50] for s in (1, 2, 3)]
		d_rows.append({"tag": tag, "note": " | ".join(reads),
			"images": [copy(f"{label}_{rep}.png", bench2[f"{label}_s{rep}"])
				for rep in (1, 2, 3)]})
	sections.append({"title": "Color every garment",
		"intro": "Uncolored garments get a consistent but arbitrary default "
			"palette; naming colors is what makes the palette yours.",
		"rows": d_rows})

	# --- old emphasis renders superseded by bench 2 above -------------------
	_ = OLD_LADDER  # earlier-session evidence kept in data/experiments/size_ladder/

	# --- genital cues ------------------------------------------------------
	cue_rows = []
	for label, tag, verdict in (
			("A0_none", "(no cue)", "0/2 rendered — featureless or thighs covering"),
			("A1_pussy", "pussy", "1/2 — a coin flip"),
			("A2_pussy_folds", "pussy, detailed realistic folds",
				"2/2 with full labia detail — the detail phrase carries the render"),
			("A3_guide_recipe", "smooth shaved pussy, detailed realistic folds",
				"1/2 — the `smooth shaved` prefix added nothing")):
		imgs = [copy(f"cue_{label}_s{rep}.png", claims[f"{label}_s{rep}"])
			for rep in (1, 2)]
		cue_rows.append({"tag": tag, "note": verdict, "images": imgs})
	sections.append({"title": "Genital cues (nude, same identity)",
		"intro": "`nude` alone draws featureless anatomy. The cue wording "
			"decides whether anything renders at all.",
		"rows": cue_rows})

	# --- two-girl separation ----------------------------------------------
	two = [copy("twogirl_asym.png", claims["C1_asym"]),
		copy("twogirl_plain.png", claims["C2_plain"])]
	sections.append({"title": "Two figures, two sizes",
		"intro": "Per-figure tags alone kept the sizes distinct — with and "
			"without `asymmetrical pose`. Keep the pose tag for variety, not "
			"separation.",
		"rows": [{"tag": "one girl with huge breasts, one girl with small breasts "
				"+ asymmetrical pose", "note": "clearly distinct sizes",
				"images": [two[0]]},
			{"tag": "one girl with huge breasts, one girl with small breasts",
				"note": "distinct without the pose tag", "images": [two[1]]}]})

	# --- bra strain --------------------------------------------------------
	bra = [copy("bra_strain.png", claims["D1_strain"]),
		copy("bra_plain.png", claims["D2_plain"])]
	sections.append({"title": "Bra strain recipe (large+)",
		"intro": "`tight lacy bra straining to contain heavy breasts, fabric "
			"stretch, visible overflow, bra digging into skin with red marks` "
			"+ negative `perfect fit`.",
		"rows": [{"tag": "strain recipe + negative perfect fit",
				"note": "overflow, tight fabric, pressure marks — verified",
				"images": [bra[0]]},
			{"tag": "plain `lacy bra` (control)", "note": "smooth perfect fit",
				"images": [bra[1]]}]})

	# 4. write the page
	cards = []
	for section in sections:
		rows_html = ""
		for row in section["rows"]:
			imgs = "".join(f'<img src="{name}" alt="{row["tag"]}">'
				for name in row["images"])
			rows_html += (f'<tr><td class="tag">{row["tag"]}</td>'
				f'<td class="imgs">{imgs}</td>'
				f'<td class="note">{row["note"]}</td></tr>')
		cards.append(f"""
<section>
<h2>{section["title"]}</h2>
<p class="intro">{section["intro"]}</p>
<table><thead><tr><th>tag</th><th>renders</th><th>finding</th></tr></thead>
<tbody>{rows_html}</tbody></table>
</section>""")

	page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>image_prompt_guide — evidence gallery</title>
<style>
:root {{ --bg:#14161a; --card:#22262e; --border:#333842; --text:#e6e6ea;
	--dim:#9aa0ab; --accent:#e8a2c8; }}
body {{ margin:0; padding:24px; background:var(--bg); color:var(--text);
	font:15px/1.5 system-ui, sans-serif; }}
h1 {{ font-size:22px; }} h1 small {{ color:var(--dim); font-weight:400; }}
h2 {{ font-size:17px; color:var(--accent); margin:28px 0 4px; }}
.intro {{ color:var(--dim); margin:4px 0 12px; max-width:70ch; }}
table {{ border-collapse:collapse; width:100%; background:var(--card);
	border:1px solid var(--border); border-radius:8px; overflow:hidden; }}
th, td {{ padding:10px 12px; text-align:left; vertical-align:top;
	border-top:1px solid var(--border); }}
thead th {{ border-top:none; color:var(--dim); font-size:12px;
	text-transform:uppercase; letter-spacing:.06em; }}
td.tag {{ font-family:ui-monospace, monospace; white-space:nowrap;
	width:1%; font-size:13px; }}
td.imgs img {{ height:220px; border-radius:6px; margin:2px 6px 2px 0;
	border:1px solid var(--border); display:inline-block; }}
td.note {{ color:var(--dim); max-width:38ch; font-size:13px; }}
</style></head><body>
<h1>image_prompt_guide — evidence gallery <small>· guide claims vs renders, 2026-10-09</small></h1>
{''.join(cards)}
<p class="intro">Raw records: data/experiments/guide_claims_20261009/ ·
older ladder: data/experiments/size_ladder/. Regenerate this page:
python scripts/build_guide_gallery.py</p>
</body></html>"""
	(GALLERY / "gallery.html").write_text(page, encoding="utf-8")
	print("[done]", GALLERY / "gallery.html", flush=True)
	print("[url] http://localhost:8080/debug-files/guide_gallery/gallery.html",
		flush=True)


if __name__ == "__main__":
	main()
