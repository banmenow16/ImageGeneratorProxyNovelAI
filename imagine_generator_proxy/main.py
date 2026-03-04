import os
import base64
import asyncio
import hashlib
import re
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


async def generate_handler(request: web.Request) -> web.Response:
	print("Received /generate request for " + request.query.get("prompt"))
	config = request.app.get("config", {})
	initial_prompt = 'score_9, score_8_up, score_7_up, score_6_up, source_anime'
	prompt = initial_prompt + ',' + request.query.get("prompt")
	if not prompt:
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
			print("Failed to remove cached image for force override:", e)

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
			return write_cache_and_prepare_response(data, char_dir, cached_file, want_raw, gens, digest, name)

		# If data is not binary, return it as text in JSON
		return prepare_nonbinary_response(name, data, gens, digest)
	except Exception as e:
		# log error server-side for easier debugging
		print("/generate handler error:", repr(e))
		traceback.print_exc()
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
                    print(f"NovelAI busy (attempt {attempt}/{max_retries}), retrying in {backoff}s:", s)
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
			print(f"Failed to read cached image ({context}):", e)
		else:
			print("Failed to read cached image:", e)
		return None


def write_cache_and_prepare_response(data, char_dir: Path, cached_file: Path, want_raw: bool, gens: dict, digest: str, name: str):
	try:
		char_dir.mkdir(parents=True, exist_ok=True)
		cached_file.write_bytes(bytes(data))
	except Exception as e:
		print("Failed to write cache file:", e)

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
	app.on_startup.append(on_startup)
	app.on_cleanup.append(on_cleanup)
	return app


if __name__ == "__main__":
	port = int(os.environ.get("PORT", "8080"))
	web.run_app(create_app(), host="0.0.0.0", port=port)

