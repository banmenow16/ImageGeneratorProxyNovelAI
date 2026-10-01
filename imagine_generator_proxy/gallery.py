"""Cache gallery: folder index and per-character image grids."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from aiohttp import web

LOGGER = logging.getLogger(__name__)


async def gallery_handler(request: web.Request) -> web.Response:
	cache_dir: Path = request.app.get("cache_dir")
	if cache_dir is None:
		return web.Response(text="Cache directory unavailable", status=500)
	folders = sorted([d for d in cache_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
	html = "<html><head><title>Image Cache Gallery</title></head><body><h1>Image Cache Folders</h1><ul>"
	for folder in folders:
		count = sum(1 for _ in folder.glob('*.png'))
		html += f'<li><a href="/folder/{folder.name}">{folder.name}</a> ({count})</li>'
	if not folders:
		html += "<li>No cached folders found.</li>"
	html += "</ul></body></html>"
	return web.Response(text=html, content_type="text/html")


async def folder_handler(request: web.Request) -> web.Response:
	character = request.match_info['character']
	cache_dir: Path = request.app.get("cache_dir")
	if cache_dir is None:
		return web.Response(text="Cache directory unavailable", status=500)
	char_dir = cache_dir / character
	if not char_dir.exists() or not char_dir.is_dir():
		return web.Response(text="Folder not found", status=404)

	# Load prompt metadata from the database if available
	metadata = {}
	try:
		conn = sqlite3.connect(str(request.app["db_path"]))
		try:
			cursor = conn.cursor()
			cursor.execute('SELECT digest, prompt, first_appearance FROM images WHERE character = ?', (character,))
			for digest, prompt, first_appearance in cursor.fetchall():
				metadata[digest] = (prompt, first_appearance)
		finally:
			conn.close()
	except Exception:
		LOGGER.exception("Failed to query image metadata for gallery")

	images = []
	for image_path in char_dir.glob('*.png'):
		digest = image_path.stem
		prompt, first_appearance = metadata.get(digest, (None, None))
		mtime = datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(sep=' ', timespec='seconds')
		date_label = first_appearance or mtime
		images.append((image_path.name, prompt, date_label, image_path.stat().st_mtime))

	images.sort(key=lambda item: item[3], reverse=True)

	html = f"""<html><head><title>{character} Images</title><style>
		body {{ font-family: Arial, sans-serif; margin: 0; padding: 16px; background: #f0f0f0; }}
		h1 {{ margin-top: 0; }}
		.page-header {{ display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 18px; }}
		.page-header a {{ text-decoration: none; color: #0066cc; }}
		.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 320px)); gap: 16px; justify-content: center; }}
		.card {{ width: 100%; max-width: 320px; background: #fff; border: 1px solid #ddd; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
		.card img {{ width: 100%; height: auto; display: block; }}
		.card-body {{ padding: 12px; }}
		.card-body p {{ margin: 0 0 8px; line-height: 1.4; word-break: break-word; }}
		.card-body .date {{ color: #666; font-size: 0.9em; }}
	</style></head><body><div class="page-header"><h1>{character} Images</h1><a href='/'>Back to Folders</a></div>"""
	if not images:
		html += "<p>No images found in this folder.</p>"
	else:
		html += '<div class="grid">'
		for filename, prompt, date_label, _ in images:
			html += f'<div class="card"><img src="/cache/{character}/{filename}" alt="{filename}"><div class="card-body">'
			if prompt:
				html += f'<p>{prompt}</p>'
			html += f'<p class="date">{date_label}</p></div></div>'
		html += '</div>'
	html += "</body></html>"
	return web.Response(text=html, content_type="text/html")
