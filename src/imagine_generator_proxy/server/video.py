"""Video pipeline HTTP layer: generation endpoint (streaming progress),
workbench page, and video gallery. Videos animate gallery images via the
registered VideoGenerator backends."""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path
from typing import Optional

from aiohttp import web

from imagine_generator_proxy.generation.video_base import VideoGenerator, VideoRequest
from imagine_generator_proxy.server.storage import ensure_db_schema

LOGGER = __import__("logging").getLogger(__name__)

PAGE_PATH = Path(__file__).parent / "video.html"


def _video_backends(app: web.Application) -> dict:
	return app.get("video_backends") or {}


async def video_page_handler(request: web.Request) -> web.Response:
	return web.Response(text=PAGE_PATH.read_text(encoding="utf-8"), content_type="text/html")


async def video_generate_stream_handler(request: web.Request) -> web.StreamResponse:
	try:
		body = await request.json()
	except Exception:
		return web.json_response({"error": "request body must be JSON"}, status=400)

	character = (body.get("character") or "default").strip() or "default"
	digest = (body.get("source_digest") or "").strip()
	source = _resolve_source(request.app, character, digest)
	if source is None:
		return web.json_response({"error": "source image not found (character + source_digest)"}, status=400)

	backends = _video_backends(request.app)
	backend_name = body.get("engine") or request.app.get("video_backend", "wan22")
	backend = backends.get(backend_name)
	if backend is not None and getattr(backend, "nsfw_blocked", False):
		return web.json_response({"error": f"{backend_name} is disabled: NSFW-blocked "
			"(empirically verified) — pick another engine"}, status=400)
	if backend is None:
		return web.json_response({"error": f"unknown video engine {backend_name!r}",
			"available": sorted(backends)}, status=400)
	try:
		duration = max(2, min(int(body.get("duration", 5)), 20))
	except (TypeError, ValueError):
		duration = 5

	response = web.StreamResponse(headers={
		"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"})
	await response.prepare(request)

	async def emit(event: dict) -> None:
		await response.write((json.dumps(event) + "\n").encode("utf-8"))

	await emit({"event": "step", "phase": "uploading", "engine": backend.display_name})
	video_request = VideoRequest(
		character=character, source_image=source, source_digest=digest,
		prompt=(body.get("prompt") or "").strip(),
		negative_prompt=(body.get("negative") or "").strip(),
		duration_seconds=duration,
		width=int(body.get("width", 832)), height=int(body.get("height", 1216)),
	)
	await emit({"event": "step", "phase": "generating", "engine": backend.display_name,
		"duration": duration})
	video = await backend.generate_video(video_request)
	if video is None:
		detail = getattr(backend, "last_error", "") or "video generation failed"
		if "insufficientBuzz" in detail:
			detail = "insufficient Buzz on your Civitai account — top up at civitai.com/buzz"
		await emit({"event": "error", "error": detail})
		await response.write_eof()
		return response

	video_id = _store(request.app, character, digest, video_request, video, backend_name)
	url = f"/video-files/{character}/v{video_id}.mp4"
	await emit({"event": "done", "video_id": video_id, "url": url,
		"bytes": len(video.data), "engine": backend.display_name})
	await response.write_eof()
	return response


def _resolve_source(app: web.Application, character: str, digest: str) -> Optional[Path]:
	if not digest:
		return None
	path = app["cache_dir"] / character / f"{digest}.png"
	return path if path.exists() else None


def _store(app, character: str, digest: str, req: VideoRequest, video, engine: str) -> int:
	videos_dir = Path(app["videos_dir"])
	run_dir = videos_dir / character
	run_dir.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(str(app["db_path"]))
	try:
		conn.execute(
			"INSERT INTO videos (character, source_digest, prompt, engine, duration_seconds,"
			" width, height, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'succeeded')",
			(character, digest, req.prompt, engine, req.duration_seconds, req.width, req.height))
		conn.commit()
		video_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
	finally:
		conn.close()
	(run_dir / f"v{video_id}.mp4").write_bytes(video.data)
	return int(video_id)


async def video_gallery_handler(request: web.Request) -> web.Response:
	conn = sqlite3.connect(str(request.app["db_path"]))
	try:
		conn.row_factory = sqlite3.Row
		rows = conn.execute(
			"SELECT id, character, source_digest, prompt, engine, duration_seconds, created_at"
			" FROM videos WHERE status='succeeded' ORDER BY id DESC LIMIT 200").fetchall()
	finally:
		conn.close()
	cards = []
	for r in rows:
		src = f"/cache/{r['character']}/{r['source_digest']}.png" if r["source_digest"] else ""
		prompt = html.escape(r["prompt"] or "")
		cards.append(
			f'<div class="card"><video controls preload="metadata" src="/video-files/{r["character"]}/v{r["id"]}.mp4"></video>'
			f'<div class="body"><div class="label">{html.escape(r["character"])} · {html.escape(r["engine"])} · {r["duration_seconds"]}s</div>'
			f'<div class="prompt">{prompt}</div>'
			f'{f"<a href=\"/video?src={r["source_digest"]}&character={r["character"]}\">↻ re-animate</a>" if src else ""}'
			f'<div class="date">{r["created_at"]}</div></div></div>')
	page = f"""<html><head><title>Videos</title><style>
	body{{margin:0;padding:20px;background:#14161a;color:#e6e9ee;font-family:'Segoe UI',sans-serif}}
	h1{{font-size:22px}} a{{color:#6ea8fe}}
	.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
	.card{{background:#22262e;border:1px solid #333842;border-radius:10px;overflow:hidden}}
	video{{width:100%;height:auto;display:block;background:#000}}
	.body{{padding:10px 12px;font-size:12px}} .label{{color:#9aa3af;margin-bottom:6px}}
	.prompt{{font-family:Consolas,monospace;font-size:11px;word-break:break-word}}
	.date{{color:#666;font-size:11px;margin-top:6px}}</style></head><body>
	<h1>Video Gallery</h1><p style="color:#9aa3af"><a href="/video">→ video workbench</a></p>
	<div class="grid">{''.join(cards) or '<p>No videos yet.</p>'}</div></body></html>"""
	return web.Response(text=page, content_type="text/html")


async def video_engines_handler(request: web.Request) -> web.Response:
	backends = _video_backends(request.app)
	engines = [{"id": b.backend_name, "display": b.display_name,
		"provider": type(b).__module__.rsplit(".", 1)[-1],
		"nsfw_verified": bool(getattr(b, "nsfw_verified", False))}
		for b in backends.values()
		if not getattr(b, "nsfw_blocked", False)]  # hidden, code kept intact
	default = request.app.get("video_backend", "wan22")
	return web.json_response({"engines": engines, "default": default})


def add_routes(app: web.Application) -> None:
	app.router.add_get("/api/video-engines", video_engines_handler)
	app.router.add_get("/video", video_page_handler)
	app.router.add_post("/video/generate/stream", video_generate_stream_handler)
	app.router.add_get("/video/gallery", video_gallery_handler)


def ensure_video_schema(conn: sqlite3.Connection) -> None:
	conn.execute('''CREATE TABLE IF NOT EXISTS videos (
		id INTEGER PRIMARY KEY,
		character TEXT,
		source_digest TEXT,
		prompt TEXT,
		engine TEXT,
		duration_seconds REAL,
		width INTEGER,
		height INTEGER,
		video_path TEXT,
		status TEXT,
		buzz_cost INTEGER,
		error TEXT,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	)''')
	conn.commit()
