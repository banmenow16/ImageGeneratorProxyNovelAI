"""Cache gallery: folder index and per-character image grids, showing both
the positive prompt and the negative prompt for each cached image."""

from __future__ import annotations

import html
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from aiohttp import web

from imagine_generator_proxy.prompts import strip_negative_leak

LOGGER = logging.getLogger(__name__)

_STYLE = """
	:root { --bg:#14161a; --card:#22262e; --border:#333842; --text:#e6e9ee;
		--muted:#9aa3af; --accent:#6ea8fe; --neg:#e7a33e; }
	* { box-sizing:border-box; }
	body { margin:0; padding:20px; background:var(--bg); color:var(--text);
		font-family:"Segoe UI", system-ui, sans-serif; }
	h1 { margin:0 0 4px; font-size:22px; }
	a { color:var(--accent); text-decoration:none; }
	.sub { color:var(--muted); font-size:13px; margin-bottom:18px; }
	.folder-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:10px; }
	.folder-card { background:var(--card); border:1px solid var(--border); border-radius:10px;
		padding:12px 14px; display:flex; justify-content:space-between; align-items:baseline; }
	.folder-card a { font-weight:600; }
	.folder-card .count { color:var(--muted); font-size:13px; }
	.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; }
	.card { background:var(--card); border:1px solid var(--border); border-radius:10px;
		overflow:hidden; display:flex; flex-direction:column; }
	.card img { width:100%; height:auto; display:block; background:#0d0f12; min-height:80px; }
	.card-body { padding:12px 14px; display:flex; flex-direction:column; gap:8px; flex:1; }
	.label { font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
	.prompt { font-family:Consolas,monospace; font-size:11.5px; line-height:1.45;
		word-break:break-word; white-space:pre-wrap; max-height:5.8em; overflow:hidden; }
	.prompt.full { max-height:none; overflow:visible; }
	.clamp-btn { background:none; border:0; color:var(--accent); font-size:11px;
		padding:2px 0; cursor:pointer; text-align:left; }
	.video-link { display:inline-block; background:rgba(231,163,62,.14); color:#e7a33e;
		border-radius:6px; padding:5px 10px; font-size:11.5px; font-weight:600; }
	.video-link:hover { background:rgba(231,163,62,.26); }
	.negative { color:var(--neg); opacity:.85; font-size:11px; }
	.run-link { display:inline-block; background:rgba(110,168,254,.12); color:var(--accent);
		border-radius:6px; padding:5px 10px; font-size:11.5px; font-weight:600; }
	.run-link:hover { background:rgba(110,168,254,.22); }
	.date { color:var(--muted); font-size:11px; margin-top:auto; }
"""

_PAGE = """<html><head><title>{title}</title><style>{style}</style></head><body>
<div class="page-header"><h1>{title}</h1>{header_extra}</div>
<div class="sub">{sub}</div>
{body}
</body></html>"""


def _esc(text: str | None) -> str:
	return html.escape(text or "", quote=True)


async def gallery_handler(request: web.Request) -> web.Response:
	cache_dir: Path | None = request.app.get("cache_dir")
	if cache_dir is None:
		return web.Response(text="Cache directory unavailable", status=500)
	folders = sorted((d for d in cache_dir.iterdir() if d.is_dir()), key=lambda d: d.name)

	cards = []
	for folder in folders:
		count = sum(1 for _ in folder.glob("*.png"))
		cards.append(
			f'<div class="folder-card"><a href="/folder/{_esc(folder.name)}">'
			f"{_esc(folder.name)}</a><span class='count'>{count}</span></div>"
		)
	body = '<div class="folder-grid">' + "".join(cards) + "</div>" if cards \
		else "<p>No cached folders found.</p>"

	page = _PAGE.format(
		title="Image Cache Gallery",
		style=_STYLE,
		header_extra="",
		sub=f"{len(folders)} character folders",
		body=body,
	)
	return web.Response(text=page, content_type="text/html")


async def folder_handler(request: web.Request) -> web.Response:
	character = request.match_info["character"]
	cache_dir: Path | None = request.app.get("cache_dir")
	if cache_dir is None:
		return web.Response(text="Cache directory unavailable", status=500)
	char_dir = cache_dir / character
	if not char_dir.exists() or not char_dir.is_dir():
		return web.Response(text="Folder not found", status=404)

	# digest -> (prompt, negative_prompt, raw_prompt, first_appearance)
	metadata: dict = {}
	try:
		conn = sqlite3.connect(str(request.app["db_path"]))
		try:
			cursor = conn.cursor()
			cursor.execute(
				"SELECT digest, prompt, negative_prompt, raw_prompt, first_appearance"
				" FROM images WHERE character = ?",
				(character,),
			)
			for digest, prompt, negative, raw_prompt, first_appearance in cursor.fetchall():
				metadata[digest] = (prompt, negative, raw_prompt, first_appearance)
		finally:
			conn.close()
	except Exception:
		LOGGER.exception("Failed to query image metadata for gallery")

	images = []
	for image_path in char_dir.glob("*.png"):
		digest = image_path.stem
		prompt, negative, raw_prompt, first_appearance = metadata.get(digest, (None, None, None, None))
		mtime = datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(sep=" ", timespec="minutes")
		images.append((image_path.name, image_path.stem, prompt, negative, raw_prompt, first_appearance or mtime, image_path.stat().st_mtime))

	images.sort(key=lambda item: item[6], reverse=True)

	cards = []
	for filename, digest, prompt, negative, raw_prompt, date_label, _ in images:
		if prompt:
			esc_prompt = _esc(prompt)
			clamp = (f'<button class="clamp-btn" data-target="p">more ▾</button>'
				if len(prompt) > 220 else "")
			prompt_html = ('<div><div class="label">prompt</div>'
				f'<div class="prompt">{esc_prompt}</div>{clamp}</div>')
		else:
			prompt_html = '<div><div class="label">prompt</div><div class="prompt">—</div></div>'
		if negative is None:
			negative_html = ('<div><div class="label">negative</div>'
				'<div class="prompt negative">(not recorded)</div></div>')
		else:
			display = negative.rstrip(", ").strip()
			negative_html = (
				f'<div><div class="label">negative</div><div class="prompt negative">{_esc(display)}</div></div>'
				if display else ""
			)
		# one click from the gallery into a pre-filled workbench run: prefer
		# the raw caller prompt with the known SillyTavern negative-leak cut
		# (raw prompts often carry 'negative: ...' inside the positive — the
		# massager strips it at generation time, so strip it here too or the
		# prefilled field reads as a bug), and carry the negative exactly as
		# the card displays it — matching the manual copy-paste workflow
		experiment_href = "/experiment?" + urlencode({
			"character": character,
			"prompt": strip_negative_leak(raw_prompt or prompt or ""),
			"negative": (negative or "").rstrip(", ").strip(),
		})
		cards.append(
			f'<div class="card"><img src="/cache/{_esc(character)}/{_esc(filename)}" alt="{_esc(filename)}">'
			f'<div class="card-body">{prompt_html}{negative_html}'
			f'<a class="run-link" href="{_esc(experiment_href)}">▶ run in experiment workbench</a>'
		f'<a class="video-link" href="/video?src={_esc(digest)}&amp;character={_esc(character)}">🎬 animate</a>'
			f'<div class="date">{_esc(date_label)}</div></div></div>'
		)

	body = '<div class="grid">' + "".join(cards) + "</div>" if images \
		else "<p>No images found in this folder.</p>"
	body += ("<script>document.addEventListener('click', e => {""const b = e.target.closest('.clamp-btn'); if (!b) return;""const p = b.previousElementSibling; p.classList.toggle('full');""b.textContent = p.classList.contains('full') ? 'less ▴' : 'more ▾';});</script>")

	page = _PAGE.format(
		title=f"{character} Images",
		style=_STYLE,
		header_extra='<a href="/">← Back to Folders</a>',
		sub=f"{len(images)} images, newest first · click an image for the full-size file",
		body=body,
	)
	return web.Response(text=page, content_type="text/html")
