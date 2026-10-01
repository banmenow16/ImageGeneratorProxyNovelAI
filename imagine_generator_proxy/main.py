import os
import asyncio
import re
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
import sys
import time
from pathlib import Path
try:
	import yaml
except Exception:
	yaml = None
from aiohttp import web, ClientSession

from novelai_api.NovelAI_API import NovelAIAPI
from novelai_api.ImagePreset import ImageModel

from imagine_generator_proxy import experiment, gallery
from imagine_generator_proxy.cache import (
	prepare_nonbinary_response,
	read_cached_image,
	write_cache_and_prepare_response,
)
from imagine_generator_proxy.generation import ImageGenerator, NovelAIGenerator, ImageGenerationType
from imagine_generator_proxy.openrouter_reviewer import OpenRouterReviewer
from imagine_generator_proxy.prompt_massager import (
	MassagedPrompt,
	PromptMassager,
	build_negative_prompt,
	legacy_assemble,
)
from imagine_generator_proxy.reviewer import NoopReviewer
from imagine_generator_proxy.storage import (
	compute_digest,
	ensure_db_schema,
	log_review_outcome,
	record_generated_image,
)

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
	raw_prompt = request.query.get("prompt")
	if not raw_prompt:
		LOGGER.warning("Bad request: /generate is missing the 'prompt' query parameter")
		return web.json_response({"error": "missing 'prompt' query parameter"}, status=400)

	# Character name (used for namespacing cache). Keep only safe characters.
	raw_character = request.query.get("character") or "default"
	character = re.sub(r"[^A-Za-z0-9_-]", "_", raw_character)

	# Cache lookup
	cache_dir: Path = request.app.get("cache_dir") or (Path(__file__).parent / "cache")
	q_negative = request.query.get("negative_prompt") or ""

	# Massage the prompt. The safety filter runs on every path: inside
	# massage() when massage mode is on, via check_safety() when bypassed.
	massager: PromptMassager = request.app["massager"]
	use_massage = massager.enabled and request.query.get("massage") not in ("0", "false")
	if use_massage:
		try:
			result = massager.massage(raw_prompt, character)
		except Exception:
			# A massager bug must never break generation: fall back to the
			# legacy assembly and the legacy (un-prefixed) digest formula.
			LOGGER.exception("PromptMassager failed; falling back to legacy prompt assembly")
			result = MassagedPrompt(original=raw_prompt, effective=legacy_assemble(raw_prompt), version=None)
	else:
		blocked = massager.check_safety(raw_prompt)
		if blocked is not None:
			result = MassagedPrompt(original=raw_prompt, effective=raw_prompt, version=None,
				blocked_rule=blocked[0], blocked_reason=blocked[1])
		else:
			result = MassagedPrompt(original=raw_prompt, effective=legacy_assemble(raw_prompt), version=None)

	if result.blocked_reason is not None:
		LOGGER.warning("Blocked prompt for character=%s rule=%s: %s", character, result.blocked_rule, result.blocked_reason)
		return web.json_response({"error": "prompt blocked by safety filter", "rule": result.blocked_rule,
			"reason": result.blocked_reason}, status=400)

	prompt = result.effective
	version = result.version
	# legacy=True keeps the historical negative blocks (and thus the legacy
	# digest) byte-identical on the bypass path
	negative_prompt = build_negative_prompt(q_negative, config, extra=result.negative_additions,
		legacy=(version is None))

	# Build cache key from character, prompt, negative_prompt and model name.
	# version None keeps the historical formula so bypass requests still hit
	# the pre-massager cache; massaged prompts are namespaced by rule version.
	model = ImageModel.Anime_v45_Full
	digest = compute_digest(character, prompt, negative_prompt, model, version)
	char_dir = cache_dir / character
	cached_file = char_dir / f"{digest}.png"

	want_raw = request.query.get("raw") in ("1", "true") or "image/" in request.headers.get("Accept", "")

	# Force override cache? (use ?force=1 or ?force=true)
	force = request.query.get("force") in ("1", "true")

	if not force:
		resp = read_cached_image(cached_file, want_raw, effective_prompt=prompt, massager_version=version)
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

		resp = read_cached_image(cached_file, want_raw, context="after wait", effective_prompt=prompt, massager_version=version)
		if resp:
			return resp

	else:
		# Register an event that signals this digest is being generated
		gen_event = asyncio.Event()
		gens[digest] = gen_event

	generator: NovelAIGenerator = request.app["generator"]

	LOGGER.info("Sending request to NovelAI. Queue size: %s", len(gens))

	try:
		image = await generator.generate(prompt, negative_prompt, prompt_is_complete=(version is not None))

		if image is None:
			# cleanup generation event so waiters don't hang
			try:
				gens[digest].set()
			except Exception:
				pass
			gens.pop(digest, None)
			return web.json_response({"error": "novelai busy, try again later"}, status=503)

		# Vision review (config-gated; NoopReviewer approves instantly).
		# Quality rejections regenerate with the reviewer's proposed tag fixes
		# (safety-checked) plus a new seed; each attempt carries the previous
		# rejection notes so the reviewer verifies the fix. Minor-category
		# rejections are fail-closed and never served.
		reviewer = request.app.get("reviewer") or NoopReviewer()
		log_reviews = bool(config.get("vision_review"))
		review_retries = int(config.get("vision_review_retries", 1))
		verdict = None
		previous_notes = ""
		attempt_prompt = prompt
		attempt_negative = negative_prompt
		for attempt in range(review_retries + 1):
			verdict = await reviewer.review(bytes(image.data), attempt_prompt,
				attempt=attempt, previous_notes=previous_notes, character=character)
			if log_reviews:
				log_review_outcome(request.app.get("db_path"), digest, character, verdict,
					attempt=attempt, tags=attempt_prompt)
			if verdict.approved:
				break
			previous_notes = verdict.notes or ""
			if verdict.category == "minor":
				LOGGER.warning("Reviewer minor-flag (notes=%s) — refusing to serve", (verdict.notes or "")[:200])
				try:
					gens[digest].set()
				except Exception:
					pass
				gens.pop(digest, None)
				return web.json_response({"error": "image rejected by age-safety review"}, status=422)
			if attempt < review_retries:
				fix_add = [t for t in verdict.fix_add if massager.check_safety(t) is None]
				fix_negate = [t for t in verdict.fix_negate if massager.check_safety(t) is None]
				if fix_add:
					attempt_prompt = attempt_prompt + ", " + ", ".join(fix_add)
					LOGGER.info("Applying reviewer fix tags: %s", ", ".join(fix_add))
				if fix_negate:
					attempt_negative = attempt_negative + ", " + ", ".join(fix_negate)
					LOGGER.info("Applying reviewer fix negatives: %s", ", ".join(fix_negate))
				LOGGER.warning("REGENERATING %s for character=%s digest=%s attempt=%d/%d "
					"(reviewer rejected on quality: %s)",
					"with fix tags" if (fix_add or fix_negate) else "with new seed",
					character, digest[:12], attempt + 2, review_retries + 1,
					(verdict.notes or "")[:200])
				image = await generator.generate(attempt_prompt, attempt_negative,
					prompt_is_complete=(version is not None))
				if image is None:
					break
		if verdict is not None and not verdict.approved:
			LOGGER.warning("Reviewer still rejects on quality after retries — serving anyway")

		# Write cache outside the lock
		if isinstance(image.data, (bytes, bytearray)):
			resp = write_cache_and_prepare_response(image.data, char_dir, cached_file, want_raw, gens, digest, image.name,
				effective_prompt=prompt, massager_version=version)
			record_generated_image(request.app["db_path"], character, digest, prompt,
				negative_prompt, str(model), raw_prompt, version)
			return resp

		# If data is not binary, return it as text in JSON
		return prepare_nonbinary_response(image.name, image.data, gens, digest)
	except Exception as e:
		# log error server-side for easier debugging
		LOGGER.exception("/generate handler error")
		try:
			gens[digest].set()
		except Exception:
			pass
		gens.pop(digest, None)
		return web.json_response({"error": str(e)}, status=500)


def load_config() -> dict:
	"""Load config.yaml next to the package (PyYAML, with a flat key:value
	fallback parser when PyYAML is unavailable)."""
	cfg_path = Path(__file__).parent / "config.yaml"
	if not cfg_path.exists():
		return {}

	try:
		text = cfg_path.read_text(encoding="utf-8")
	except Exception:
		LOGGER.exception("Failed to read config.yaml")
		return {}

	if yaml:
		try:
			return yaml.safe_load(text) or {}
		except Exception:
			LOGGER.exception("Failed to parse config.yaml with PyYAML")
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
	LOGGER.warning("PyYAML not installed — using simple config parser")
	return cfg


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
	except Exception:
		LOGGER.exception("Failed to create cache directory")
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
	ensure_db_schema(conn)
	conn.close()
	app["db_path"] = db_path

	# Load YAML config (if present) and authenticate
	config = load_config()
	app["config"] = config

	# Prompt massager (safety filter + cleanup + enhance rules)
	app["massager"] = PromptMassager.from_config(config, Path(__file__).parent)

	# Generation wrapper (lock + retry/backoff) around the NovelAI API
	app["generator"] = NovelAIGenerator(api, config, lock=app["generate_lock"])

	# Post-generation image reviewer: two-tier OpenRouter vision review when
	# enabled in config (images are sent to OpenRouter); noop otherwise.
	if config.get("vision_review"):
		app["reviewer"] = OpenRouterReviewer.from_config(config, db_path=app["db_path"]) or NoopReviewer()
	else:
		app["reviewer"] = NoopReviewer()

	# Experiment workbench reviewer: runs whenever an OpenRouter key exists,
	# independently of the production vision_review flag.
	app["experiment_reviewer"] = experiment.build_experiment_reviewer(config, app["db_path"])

	token = config.get("token") or os.environ.get("NOVELAI_TOKEN")
	email = os.environ.get("NOVELAI_EMAIL")
	password = os.environ.get("NOVELAI_PASSWORD")

	if token:
		try:
			await api.high_level.login_with_token(token)
		except Exception:
			LOGGER.exception("Failed to login with token")
	elif email and password:
		try:
			await api.high_level.login(email, password)
		except Exception:
			LOGGER.exception("Failed to login with email/password")


async def on_cleanup(app: web.Application):
	session: ClientSession = app.get("session")
	if session:
		await session.close()


def create_app() -> web.Application:
	app = web.Application()
	app.router.add_get("/generate", generate_handler)
	app.router.add_get("/", gallery.gallery_handler)
	app.router.add_get("/folder/{character}", gallery.folder_handler)
	app.router.add_static('/cache', str(Path(__file__).parent / "cache"))
	experiment.add_routes(app)
	app.on_startup.append(on_startup)
	app.on_cleanup.append(on_cleanup)
	return app


if __name__ == "__main__":
	from imagine_generator_proxy.__main__ import main
	main()

