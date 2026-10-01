import os
import base64
import asyncio
import hashlib
import re
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
import sys
import time
from datetime import datetime
from pathlib import Path
import traceback
try:
	import yaml
except Exception:
	yaml = None
from aiohttp import web, ClientSession, TCPConnector
from aiohttp.resolver import AsyncResolver

from novelai_api.NovelAI_API import NovelAIAPI
from novelai_api.ImagePreset import ImageModel, ImagePreset, ImageGenerationType

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
	if logging.getLogger().handlers:
		return

	log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
	log_dir = Path(__file__).parent / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(formatter)
	file_handler = RotatingFileHandler(
		log_dir / "server.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
	)
	file_handler.setFormatter(formatter)

	root_logger = logging.getLogger()
	root_logger.setLevel(getattr(logging, log_level, logging.INFO))
	root_logger.addHandler(console_handler)
	root_logger.addHandler(file_handler)


@web.middleware
async def request_logging_middleware(request: web.Request, handler):
	started = time.perf_counter()
	try:
		response = await handler(request)
	except Exception:
		elapsed = time.perf_counter() - started
		LOGGER.exception(
			"Unhandled exception while serving %s %s (%.3fs)",
			request.method,
			request.path,
			elapsed,
		)
		raise

	elapsed = time.perf_counter() - started
	if response.status >= 400:
		LOGGER.warning(
			"HTTP %s response for %s %s from %s (%.3fs)",
			response.status,
			request.method,
			request.path,
			request.remote or "unknown client",
			elapsed,
		)
	else:
		LOGGER.info(
			"HTTP %s %s -> %s (%.3fs)",
			request.method,
			request.path,
			response.status,
			elapsed,
		)
	return response


async def generate_handler(request: web.Request) -> web.Response:
	config = request.app.get("config", {})
	initial_prompt = 'score_9, score_8_up, score_7_up, score_6_up, source_anime'
	prompt = initial_prompt + ',' + request.query.get("prompt")
	if not prompt:
		LOGGER.warning("Bad request: /generate is missing the 'prompt' query parameter")
		return web.json_response({"error": "missing 'prompt' query parameter"}, status=400)

	# Character name (used for namespacing cache). Keep only safe characters.
	raw_character = request.query.get("character") or "default"
	character = re.sub(r"[^A-Za-z0-9_-]", "_", raw_character)

	# Cache lookup
	cache_dir: Path = request.app.get("cache_dir") or (Path(__file__).parent / "cache")
	# Load negative prompt: prefer query parameter `negative_prompt` if provided,
	# otherwise fall back to config value or the built-in default.
	q_negative = request.query.get("negative_prompt")
	if(q_negative is None):
		q_negative = ''  # Default empty if not provided
	negative_prompt1 = 'text, watermark, signature, letters, writing, logo, advertisement, caption, error, emoji, chat, discord, message'
	negative_prompt2 = 'deformed, ugly, blurry, lowres, extra limbs, mutated hands, poor anatomy'
	negative_prompt3 = 'bad lighting, overexposed, underexposed, worst quality, low quality, jpeg artifacts, ugly, duplicate'
	negative_prompt4 = 'unnatural proportions'
	negative_prompt = ', '.join([negative_prompt1, negative_prompt2, negative_prompt3, negative_prompt4, q_negative])


	# Build cache key from character, prompt, negative_prompt and model name
	model = ImageModel.Anime_v45_Full
	key_material = "|".join([character, prompt, negative_prompt, str(model)])
	digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
	char_dir = cache_dir / character
	cached_file = char_dir / f"{digest}.png"

	want_raw = request.query.get("raw") in ("1", "true") or "image/" in request.headers.get("Accept", "")

	# Force override cache? (use ?force=1 or ?force=true)
	force = request.query.get("force") in ("1", "true")

	if not force:
		resp = read_cached_image(cached_file, want_raw)
		if resp:
			return resp

	# If forcing cache override, remove existing cached file so generation proceeds
	if force and cached_file.exists():
		try:
			cached_file.unlink()
		except Exception as e:
			LOGGER.exception("Failed to remove cached image for force override")

	# Per-key generation events to coordinate concurrent requests for same prompt
	gens: dict = request.app.setdefault("generations", {})
	gen_event = gens.get(digest)
	if gen_event is not None:
		# Someone else is generating this image; wait for them to finish and return cached result
		try:
			await gen_event.wait()
		except Exception:
			pass

		resp = read_cached_image(cached_file, want_raw, context="after wait")
		if resp:
			return resp

	else:
		# Register an event that signals this digest is being generated
		gen_event = asyncio.Event()
		gens[digest] = gen_event

	api: NovelAIAPI = request.app["novelai_api"]

	preset = ImagePreset.from_default_config(model)

	# Use global generate lock to ensure only one request hits NovelAI at a time
	lock: asyncio.Lock = request.app.get("generate_lock")
	if lock is None:
		# fallback: create one (should normally be created on startup)
		lock = asyncio.Lock()

	LOGGER.info("Sending request to NovelAI. Queue size: %s", len(gens))

	try:
		name, data = await novelai_generate(api, prompt, model, preset, negative_prompt, lock, config)

		if name is None and data is None:
			# cleanup generation event so waiters don't hang
			try:
				gens[digest].set()
			except Exception:
				pass
			gens.pop(digest, None)
			return web.json_response({"error": "novelai busy, try again later"}, status=503)

		# Write cache outside the lock
		if isinstance(data, (bytes, bytearray)):
			resp = write_cache_and_prepare_response(data, char_dir, cached_file, want_raw, gens, digest, name)
			# Store in database
			conn = sqlite3.connect(str(request.app["db_path"]))
			try:
				conn.execute('INSERT OR IGNORE INTO images (character, digest, prompt, negative_prompt, model) VALUES (?, ?, ?, ?, ?)',
					(character, digest, prompt, negative_prompt, str(model)))
				conn.commit()
			except Exception as e:
				LOGGER.exception("Failed to store generated image in database")
			finally:
				conn.close()
			return resp

		# If data is not binary, return it as text in JSON
		return prepare_nonbinary_response(name, data, gens, digest)
	except Exception as e:
		# log error server-side for easier debugging
		LOGGER.exception("/generate handler error")
		try:
			gens[digest].set()
		except Exception:
			pass
		gens.pop(digest, None)
		return web.json_response({"error": str(e)}, status=500)


async def novelai_generate(api: NovelAIAPI, prompt: str, model, preset: ImagePreset, negative_prompt: str, lock: asyncio.Lock, config: dict):
	"""Generate an image via NovelAI API using the provided lock and retry/backoff settings.

	Returns a tuple (name, data) on success, or (None, None) when NovelAI remains busy after retries.
	"""
	max_retries = int(config.get("generation_retries", 5))
	backoff = float(config.get("generation_backoff", 1.0))
	attempt = 0
	generated = False
	name = None
	data = None

	async with lock:
		while attempt < max_retries:
			try:
				async for pair in api.high_level.generate_image(prompt, model, preset, negative_prompt=negative_prompt):
					try:
						name, data = pair
					except Exception:
						name = "image.png"
						data = pair

					generated = True
					break

				if generated:
					break

			except Exception as e:
				s = str(e)
				if "429" in s or "Concurrent generation" in s or "concurrent generation" in s.lower():
					attempt += 1
					LOGGER.warning("NovelAI busy (attempt %s/%s), retrying in %ss: %s", attempt, max_retries, backoff, s)
					await asyncio.sleep(backoff)
					backoff *= 2
					continue
				raise

	if not generated:
		return None, None

	return name, data


def read_cached_image(cached_file: Path, want_raw: bool, context: str = None):
	"""Return a web Response or JSON response for the cached file, or None if not available.

	`context` is used only to customize log messages when read fails.
	"""
	try:
		if not cached_file.exists():
			return None

		data = cached_file.read_bytes()
		if want_raw:
			return web.Response(body=bytes(data), content_type="image/png")

		img_b64 = base64.b64encode(data).decode("utf-8")
		return web.json_response({"name": cached_file.name, "image_b64": img_b64, "cached": True})
	except Exception as e:
		if context:
			LOGGER.exception("Failed to read cached image (%s)", context)
		else:
			LOGGER.exception("Failed to read cached image")
		return None


def write_cache_and_prepare_response(data, char_dir: Path, cached_file: Path, want_raw: bool, gens: dict, digest: str, name: str):
	try:
		char_dir.mkdir(parents=True, exist_ok=True)
		cached_file.write_bytes(bytes(data))
	except Exception as e:
		LOGGER.exception("Failed to write cache file")

	# signal waiting requests before returning
	try:
		gens[digest].set()
	except Exception:
		pass
	gens.pop(digest, None)

	if want_raw:
		return web.Response(body=bytes(data), content_type="image/png")

	img_b64 = base64.b64encode(data).decode("utf-8")
	return web.json_response({"name": name, "image_b64": img_b64, "cached": False})


def prepare_nonbinary_response(name: str, data, gens: dict, digest: str):
	try:
		gens[digest].set()
	except Exception:
		pass
	gens.pop(digest, None)
	return web.json_response({"name": name, "image": str(data)})


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
	conn = sqlite3.connect(str(request.app["db_path"]))
	try:
		cursor = conn.cursor()
		cursor.execute('SELECT digest, prompt, first_appearance FROM images WHERE character = ?', (character,))
		for digest, prompt, first_appearance in cursor.fetchall():
			metadata[digest] = (prompt, first_appearance)
	except Exception as e:
		print("Failed to query database:", e)
	finally:
		conn.close()

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


async def on_startup(app: web.Application):
	# Use a custom DNS resolver pointing to the requested internal DNS server
	session = ClientSession()
	app["session"] = session

	api = NovelAIAPI(session)
	app["novelai_api"] = api

	# Global lock to ensure only one request calls NovelAI at a time
	app["generate_lock"] = asyncio.Lock()

	# Map of digest -> asyncio.Event for coordinating per-key generation waiters
	app["generations"] = {}

	# Prepare cache directory
	cache_dir = Path(__file__).parent / "cache"
	try:
		cache_dir.mkdir(parents=True, exist_ok=True)
	except Exception as e:
		print("Failed to create cache directory:", e)
	app["cache_dir"] = cache_dir

	# Initialize database
	db_path = Path(__file__).parent / "images.db"
	conn = sqlite3.connect(str(db_path))
	conn.execute('''CREATE TABLE IF NOT EXISTS images (
		id INTEGER PRIMARY KEY,
		character TEXT,
		digest TEXT UNIQUE,
		prompt TEXT,
		negative_prompt TEXT,
		model TEXT,
		first_appearance TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	)''')
	conn.close()
	app["db_path"] = db_path

	# Load YAML config (if present) and authenticate
	def _load_config() -> dict:
		cfg_path = Path(__file__).parent / "config.yaml"
		if not cfg_path.exists():
			return {}

		try:
			with cfg_path.open("r", encoding="utf-8") as f:
				text = f.read()
		except Exception as e:
			print("Failed to read config.yaml:", e)
			return {}

		if yaml:
			try:
				return yaml.safe_load(text) or {}
			except Exception as e:
				print("Failed to parse config.yaml with PyYAML:", e)
				return {}

		# Fallback simple parser for key: value pairs when PyYAML is not installed
		cfg = {}
		for line in text.splitlines():
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			if ":" not in line:
				continue
			k, v = line.split(":", 1)
			k = k.strip()
			v = v.strip()
			if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
				v = v[1:-1]
			cfg[k] = v
		print("PyYAML not installed — using simple config parser. To remove this message, install PyYAML.")
		return cfg

	config = _load_config()
	app["config"] = config

	token = config.get("token") or os.environ.get("NOVELAI_TOKEN")
	email = os.environ.get("NOVELAI_EMAIL")
	password = os.environ.get("NOVELAI_PASSWORD")

	if token:
		try:
			await api.high_level.login_with_token(token)
		except Exception as e:
			print("Failed to login with token:", repr(e))
			traceback.print_exc()
	elif email and password:
		try:
			await api.high_level.login(email, password)
		except Exception as e:
			print("Failed to login with email/password:", repr(e))
			traceback.print_exc()


async def on_cleanup(app: web.Application):
	session: ClientSession = app.get("session")
	if session:
		await session.close()


def create_app() -> web.Application:
	app = web.Application()
	app.router.add_get("/generate", generate_handler)
	app.router.add_get("/", gallery_handler)
	app.router.add_get("/folder/{character}", folder_handler)
	app.router.add_static('/cache', str(Path(__file__).parent / "cache"))
	app.on_startup.append(on_startup)
	app.on_cleanup.append(on_cleanup)
	return app


if __name__ == "__main__":
	port = int(os.environ.get("PORT", "8080"))
	web.run_app(create_app(), host="0.0.0.0", port=port)

